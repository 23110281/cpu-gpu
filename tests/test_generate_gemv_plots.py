"""Standalone test for generate_gemv_plots.py against a synthetic CSV.
No GPU/CUDA dependency -- pure pandas/matplotlib.
Run: <venv>/bin/python3 test_generate_gemv_plots.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "results"))
import generate_gemv_plots as ggp

HEADER = ("tag,ts,host,cudart,mode,precision,gpus,S,iters,warmup,"
          "agg_gbps,per_gpu_gbps,pct_peak_bw,wall_ms,"
          "sm_mhz_avg,sm_mhz_max,mem_mhz_avg,power_w_avg,power_w_max,temp_c_max,throttle,util_pct_avg,"
          "mem_gb_per_gpu,validated,max_rel_err,"
          "h2d_ms,d2h_ms,compute_ms,e2e_ms,transfer_pct,agg_tflops,"
          "engine,density,nnz,status\n")

ROWS = [
    # Dense rows: split mode, fp32 precision
    "split,1000,host,12090,split,fp32,1,1024,10,3,900,900,57.9,1.1,1095,1095,1215,55,55,25,none,10,0.004,1,0.001,0.1,0.05,0.02,0.17,88,4.2,dense,1.0,0,ok\n",
    "split,1001,host,12090,split,fp32,2,1024,10,3,1600,800,51.4,0.6,1095,1095,1215,55,55,25,none,10,0.004,1,0.001,0.1,0.05,0.02,0.17,88,4.2,dense,1.0,0,ok\n",
    # Dense rows: split mode, fp64 precision (to test multi-precision grid)
    "split,1002,host,12090,split,fp64,1,1024,10,3,850,850,54.7,1.2,1095,1095,1215,56,56,26,none,9,0.008,1,0.001,0.1,0.05,0.02,0.17,88,4.0,dense,1.0,0,ok\n",
    "split,1003,host,12090,split,fp64,2,1024,10,3,1500,750,48.4,0.7,1095,1095,1215,56,56,26,none,9,0.008,1,0.001,0.1,0.05,0.02,0.17,88,4.0,dense,1.0,0,ok\n",
    # Dense rows: replicas mode, fp32 precision (to test multi-mode grid)
    "split,1004,host,12090,replicas,fp32,1,1024,10,3,920,920,59.1,1.1,1095,1095,1215,54,54,24,none,11,0.004,1,0.001,0.1,0.05,0.02,0.17,88,4.3,dense,1.0,0,ok\n",
    "split,1005,host,12090,replicas,fp32,2,1024,10,3,1620,810,52.2,0.6,1095,1095,1215,54,54,24,none,11,0.004,1,0.001,0.1,0.05,0.02,0.17,88,4.3,dense,1.0,0,ok\n",
    # Dense rows: replicas mode, fp64 precision (to test full multi-mode/precision grid)
    "split,1006,host,12090,replicas,fp64,1,1024,10,3,880,880,56.6,1.2,1095,1095,1215,55,55,25,none,10,0.008,1,0.001,0.1,0.05,0.02,0.17,88,4.1,dense,1.0,0,ok\n",
    "split,1007,host,12090,replicas,fp64,2,1024,10,3,1550,775,49.7,0.7,1095,1095,1215,55,55,25,none,10,0.008,1,0.001,0.1,0.05,0.02,0.17,88,4.1,dense,1.0,0,ok\n",
    # Sparse rows: split mode, fp32 precision
    "sparse,1008,host,12090,split,fp32,1,1024,10,3,50,50,3.2,20,1095,1095,1215,55,55,25,none,5,0.001,1,0.001,0.1,0.05,0.02,0.17,88,0.1,sparse,0.05,52224,ok\n",
    # Sparse rows: split mode, fp64 precision (to test sparse multi-precision)
    "sparse,1009,host,12090,split,fp64,1,1024,10,3,48,48,3.1,21,1095,1095,1215,56,56,26,none,5,0.002,1,0.001,0.1,0.05,0.02,0.17,88,0.09,sparse,0.05,52224,ok\n",
    # Sparse rows: replicas mode, fp32 precision (to test sparse multi-mode)
    "sparse,1010,host,12090,replicas,fp32,1,1024,10,3,52,52,3.3,19,1095,1095,1215,54,54,24,none,6,0.001,1,0.001,0.1,0.05,0.02,0.17,88,0.11,sparse,0.05,52224,ok\n",
    # Sparse rows: replicas mode, fp64 precision
    "sparse,1011,host,12090,replicas,fp64,1,1024,10,3,50,50,3.2,20,1095,1095,1215,55,55,25,none,5,0.002,1,0.001,0.1,0.05,0.02,0.17,88,0.10,sparse,0.05,52224,ok\n",
]

def test_loads_dense_and_sparse_separately():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(HEADER)
        f.writelines(ROWS)
        path = f.name
    try:
        ddf = ggp.load_gemv_data(path, engine="dense")
        sdf = ggp.load_gemv_data(path, engine="sparse")
        assert len(ddf) == 8, f"expected 8 dense rows, got {len(ddf)}"
        assert len(sdf) == 4, f"expected 4 sparse rows, got {len(sdf)}"
        assert (ddf["engine"] == "dense").all()
        assert (sdf["engine"] == "sparse").all()
    finally:
        os.unlink(path)
    print("test_loads_dense_and_sparse_separately: PASS")

def test_plots_generate_without_error():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(HEADER)
        f.writelines(ROWS)
        path = f.name
    outdir = tempfile.mkdtemp()
    try:
        ggp.set_style()

        # Test dense plots
        ddf = ggp.load_gemv_data(path, engine="dense")
        ggp.plot_gbps_heatmap(ddf, outdir, "fig1_gbps_heatmap_test")
        ggp.plot_gbps_vs_size(ddf, outdir, suffix="_test")
        ggp.plot_power_util(ddf, outdir, suffix="_test")
        ggp.plot_tflops_heatmap(ddf, outdir, "fig4_tflops_heatmap_test")

        # Assert dense output files exist
        dense_files = [
            "fig1_gbps_heatmap_test.png",
            "fig2_gbps_vs_size_split_test.png",
            "fig2_gbps_vs_size_replicas_test.png",
            "fig3_power_util_split_test.png",
            "fig3_power_util_replicas_test.png",
            "fig4_tflops_heatmap_test.png",
        ]
        for fname in dense_files:
            fpath = os.path.join(outdir, fname)
            assert os.path.exists(fpath), f"Dense plot file missing: {fname}"

        # Test sparse plots
        sdf = ggp.load_gemv_data(path, engine="sparse")
        ggp.plot_gbps_heatmap(sdf, outdir, "fig1_gbps_heatmap_sparse_test")
        ggp.plot_gbps_vs_size(sdf, outdir, suffix="_sparse_test")
        ggp.plot_power_util(sdf, outdir, suffix="_sparse_test")
        ggp.plot_tflops_heatmap(sdf, outdir, "fig4_tflops_heatmap_sparse_test")

        # Assert sparse output files exist
        sparse_files = [
            "fig1_gbps_heatmap_sparse_test.png",
            "fig2_gbps_vs_size_split_sparse_test.png",
            "fig2_gbps_vs_size_replicas_sparse_test.png",
            "fig3_power_util_split_sparse_test.png",
            "fig3_power_util_replicas_sparse_test.png",
            "fig4_tflops_heatmap_sparse_test.png",
        ]
        for fname in sparse_files:
            fpath = os.path.join(outdir, fname)
            assert os.path.exists(fpath), f"Sparse plot file missing: {fname}"
    finally:
        os.unlink(path)
    print("test_plots_generate_without_error: PASS")

if __name__ == "__main__":
    test_loads_dense_and_sparse_separately()
    test_plots_generate_without_error()
    print("ALL PASS")
