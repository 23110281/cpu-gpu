/* tests/test_cpu_telemetry.c
 * Standalone unit test for src/cpu/cpu_telemetry.h.
 *
 * Build: gcc -std=c11 -I ../src -o /tmp/test_cpu_telemetry test_cpu_telemetry.c -lm
 * Run:   /tmp/test_cpu_telemetry
 */

#include "../../src/cpu/cpu_telemetry.h"
#include <stdio.h>
#include <assert.h>

int main(void) {
    printf("==> Running test_cpu_telemetry...\n");

    cpu_telemetry_t telem;
    int rc = cpu_telemetry_init(&telem);
    assert(rc == 0);

    printf("    Telemetry initialized. RAPL available: %s (Found %d packages)\n",
           telem.rapl_available ? "YES" : "NO (Non-fatal, VMs/permissions)",
           telem.num_packages);

    cpu_telemetry_start(&telem);

    /* Busy sleep ~100ms */
    struct timespec req = {0, 100000000L};
    nanosleep(&req, NULL);

    cpu_telemetry_stop(&telem);

    printf("    Elapsed time: %.3f ms\n", telem.elapsed_ms);
    printf("    Average power: %.2f W\n", telem.avg_power_watts);
    printf("    Average core clock: %.1f MHz\n", telem.avg_freq_mhz);
    printf("    Peak core temp: %.1f C\n", telem.max_temp_c);

    assert(telem.elapsed_sec > 0.05); /* at least 50ms */
    assert(telem.avg_power_watts >= 0.0);
    assert(telem.avg_freq_mhz >= 0.0);
    assert(telem.max_temp_c >= 0.0);

    printf("test_cpu_telemetry: PASS\n");
    return 0;
}
