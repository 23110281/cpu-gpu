/* torch_c2c_allocator.cu -- PyTorch CUDAPluggableAllocator with Memory Pooling
 * for NVIDIA Grace Hopper (GH200) C2C NUMA Modes.
 *
 * Implements the standard PyTorch CUDAPluggableAllocator ABI:
 *   void* malloc_fn(ssize_t size, int device, cudaStream_t stream);
 *   void  free_fn(void* ptr, ssize_t size, int device, cudaStream_t stream);
 *
 * Includes a lightweight caching pool so intermediate activation tensors in PyTorch
 * layers (SwiGLU, SDPA, Linear) reuse warm memory blocks instead of constantly
 * invoking the Linux UVM kernel driver and triggering cold page-fault storms on warm passes.
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <unistd.h>
#include <vector>
#include <mutex>
#include <map>

struct Block {
    void* ptr;
    size_t size;
};

class C2CMemoryPool {
private:
    std::mutex mtx;
    std::multimap<size_t, void*> free_blocks;
    std::map<void*, size_t> allocated_blocks;
    bool is_um;

public:
    C2CMemoryPool(bool um_mode) : is_um(um_mode) {}

    void* allocate(size_t size) {
        if (size == 0) return nullptr;
        // Align to 256 bytes for Hopper tensor core alignments
        size = (size + 255) & ~255;

        std::lock_guard<std::mutex> lock(mtx);
        auto it = free_blocks.lower_bound(size);
        if (it != free_blocks.end() && it->first <= size * 2) {
            void* p = it->second;
            size_t blk_size = it->first;
            free_blocks.erase(it);
            allocated_blocks[p] = blk_size;
            return p;
        }

        void* ptr = nullptr;
        cudaError_t err;
        if (is_um) {
            err = cudaMallocManaged(&ptr, size, cudaMemAttachGlobal);
        } else {
            err = cudaHostAlloc(&ptr, size, cudaHostAllocMapped);
        }

        if (err != cudaSuccess || !ptr) {
            // If allocation failed, try evicting cached free blocks
            for (auto fit = free_blocks.begin(); fit != free_blocks.end(); ++fit) {
                if (is_um) cudaFree(fit->second);
                else cudaFreeHost(fit->second);
            }
            free_blocks.clear();

            if (is_um) err = cudaMallocManaged(&ptr, size, cudaMemAttachGlobal);
            else err = cudaHostAlloc(&ptr, size, cudaHostAllocMapped);

            if (err != cudaSuccess) return nullptr;
        }

        allocated_blocks[ptr] = size;
        return ptr;
    }

    void deallocate(void* ptr) {
        if (!ptr) return;
        std::lock_guard<std::mutex> lock(mtx);
        auto it = allocated_blocks.find(ptr);
        if (it != allocated_blocks.end()) {
            size_t size = it->second;
            allocated_blocks.erase(it);
            free_blocks.insert({size, ptr});
        } else {
            if (is_um) cudaFree(ptr);
            else cudaFreeHost(ptr);
        }
    }
};

static C2CMemoryPool g_zero_copy_pool(false);
static C2CMemoryPool g_um_pool(true);

extern "C" {

void* c2c_zero_copy_malloc(ssize_t size, int device, cudaStream_t stream) {
    (void)device; (void)stream;
    return g_zero_copy_pool.allocate((size_t)size);
}

void c2c_zero_copy_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
    (void)size; (void)device; (void)stream;
    g_zero_copy_pool.deallocate(ptr);
}

void* c2c_um_malloc(ssize_t size, int device, cudaStream_t stream) {
    (void)device; (void)stream;
    return g_um_pool.allocate((size_t)size);
}

void c2c_um_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
    (void)size; (void)device; (void)stream;
    g_um_pool.deallocate(ptr);
}

}
