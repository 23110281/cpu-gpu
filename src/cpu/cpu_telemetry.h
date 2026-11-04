/* src/cpu/cpu_telemetry.h
 *
 * Lightweight, header-only hardware telemetry reader for Intel Xeon CPUs.
 *
 * Provides:
 *  1. Intel RAPL (Running Average Power Limit) package energy reading
 *     via Linux powercap: /sys/class/powercap/intel-rapl/intel-rapl:X/energy_uj
 *     Computes average power consumption in Watts over the timed interval.
 *  2. CPU Core Frequency telemetry (average active MHz across cores).
 *  3. Package/Core Temperature telemetry (peak degrees Celsius).
 *  4. Monotonic high-resolution wall-clock timing (clock_gettime CLOCK_MONOTONIC_RAW).
 *
 * Safe & non-crashing: If powercap is not readable or running in a VM,
 * it sets rapl_available = 0 and reports 0.0 W without throwing errors.
 */

#pragma once

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <dirent.h>
#include <fcntl.h>
#include <errno.h>

#define CPU_TELEM_MAX_PACKAGES 8
#define AMD_MSR_PWR_UNIT       0xC0010061
#define AMD_MSR_PKG_ENERGY     0xC0010299
#define INTEL_MSR_PWR_UNIT     0x00000606
#define INTEL_MSR_PKG_ENERGY   0x00000611

typedef struct {
    char energy_path[256];
    char max_range_path[256];
    uint64_t max_range_uj;
    uint64_t start_uj;
    uint64_t end_uj;
    int is_msr;
    int msr_cpu;
    double msr_energy_unit_j;
    uint32_t msr_start_ticks;
    uint32_t msr_end_ticks;
} cpu_rapl_pkg_t;

typedef struct {
    int rapl_available;
    int num_packages;
    cpu_rapl_pkg_t pkgs[CPU_TELEM_MAX_PACKAGES];

    struct timespec start_time;
    struct timespec end_time;

    double elapsed_sec;
    double elapsed_ms;
    double avg_power_watts;
    double avg_freq_mhz;
    double max_temp_c;
} cpu_telemetry_t;

/* Internal helper: Read uint64_t from a single-line sysfs file */
static inline int cpu_telem_read_u64(const char *path, uint64_t *val) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    if (fscanf(f, "%llu", (unsigned long long *)val) != 1) {
        fclose(f);
        return -1;
    }
    fclose(f);
    return 0;
}

/* Internal helper: Read 64-bit MSR register from /dev/cpu/<cpu>/msr */
static inline int cpu_telem_read_msr(int cpu, uint32_t reg, uint64_t *val) {
    char msr_path[64];
    snprintf(msr_path, sizeof(msr_path), "/dev/cpu/%d/msr", cpu);
    int fd = open(msr_path, O_RDONLY);
    if (fd < 0) return -1;
    uint64_t data = 0;
    ssize_t bytes = pread(fd, &data, sizeof(data), (off_t)reg);
    close(fd);
    if (bytes != (ssize_t)sizeof(data)) {
        return -1;
    }
    *val = data;
    return 0;
}

/* Internal helper: Read average frequency in MHz across all active online CPUs */
static inline double cpu_telem_read_avg_freq_mhz(void) {
    long num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (num_cpus <= 0) num_cpus = 1;

    double total_khz = 0.0;
    int counted = 0;
    char path[128];

    for (long i = 0; i < num_cpus; i++) {
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%ld/cpufreq/scaling_cur_freq", i);
        uint64_t khz = 0;
        if (cpu_telem_read_u64(path, &khz) == 0 && khz > 0) {
            total_khz += (double)khz;
            counted++;
        }
    }

    if (counted > 0) {
        return (total_khz / (double)counted) / 1000.0; /* kHz -> MHz */
    }

    /* Fallback: parse /proc/cpuinfo for "cpu MHz" */
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f) return 0.0;

    char line[256];
    double mhz_sum = 0.0;
    int mhz_count = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "cpu MHz", 7) == 0) {
            char *colon = strchr(line, ':');
            if (colon) {
                mhz_sum += atof(colon + 1);
                mhz_count++;
            }
        }
    }
    fclose(f);

    return (mhz_count > 0) ? (mhz_sum / (double)mhz_count) : 0.0;
}

/* Internal helper: Scan thermal zones or coretemp for max CPU temperature (°C) */
static inline double cpu_telem_read_max_temp_c(void) {
    double max_c = 0.0;
    char path[128];

    /* Probe thermal_zone0 .. thermal_zone7 */
    for (int i = 0; i < 8; i++) {
        snprintf(path, sizeof(path), "/sys/class/thermal/thermal_zone%d/temp", i);
        uint64_t raw_mc = 0;
        if (cpu_telem_read_u64(path, &raw_mc) == 0 && raw_mc > 0) {
            double c = (double)raw_mc / 1000.0;
            if (c > max_c && c < 150.0) { /* filter bogus readings */
                max_c = c;
            }
        }
    }

    /* Also check hwmon coretemp / k10temp entries if thermal zones returned 0 */
    if (max_c <= 0.0) {
        for (int h = 0; h < 8; h++) {
            for (int t = 1; t <= 16; t++) {
                snprintf(path, sizeof(path), "/sys/class/hwmon/hwmon%d/temp%d_input", h, t);
                uint64_t raw_mc = 0;
                if (cpu_telem_read_u64(path, &raw_mc) == 0 && raw_mc > 0) {
                    double c = (double)raw_mc / 1000.0;
                    if (c > max_c && c < 150.0) {
                        max_c = c;
                    }
                }
            }
        }
    }

    return max_c;
}

/* Initialize telemetry context by discovering Intel RAPL, AMD RAPL, or direct MSRs */
static inline int cpu_telemetry_init(cpu_telemetry_t *t) {
    if (!t) return -1;
    memset(t, 0, sizeof(*t));

    t->rapl_available = 0;
    t->num_packages = 0;

    /* 1. Scan /sys/class/powercap/intel-rapl for intel-rapl:0, intel-rapl:1, etc. */
    for (int pkg = 0; pkg < CPU_TELEM_MAX_PACKAGES; pkg++) {
        char energy_file[256];
        char range_file[256];
        snprintf(energy_file, sizeof(energy_file),
                 "/sys/class/powercap/intel-rapl/intel-rapl:%d/energy_uj", pkg);
        snprintf(range_file, sizeof(range_file),
                 "/sys/class/powercap/intel-rapl/intel-rapl:%d/max_energy_range_uj", pkg);

        uint64_t test_val = 0;
        if (cpu_telem_read_u64(energy_file, &test_val) == 0) {
            snprintf(t->pkgs[t->num_packages].energy_path, sizeof(t->pkgs[t->num_packages].energy_path), "%s", energy_file);
            snprintf(t->pkgs[t->num_packages].max_range_path, sizeof(t->pkgs[t->num_packages].max_range_path), "%s", range_file);

            uint64_t max_range = 0;
            if (cpu_telem_read_u64(range_file, &max_range) == 0) {
                t->pkgs[t->num_packages].max_range_uj = max_range;
            } else {
                t->pkgs[t->num_packages].max_range_uj = 0;
            }
            t->num_packages++;
        }
    }

    /* 2. Scan /sys/class/powercap/amd-rapl (AMD EPYC via powercap driver) */
    if (t->num_packages == 0) {
        for (int pkg = 0; pkg < CPU_TELEM_MAX_PACKAGES; pkg++) {
            char energy_file[256];
            char range_file[256];
            snprintf(energy_file, sizeof(energy_file),
                     "/sys/class/powercap/amd-rapl/amd-rapl:%d/energy_uj", pkg);
            snprintf(range_file, sizeof(range_file),
                     "/sys/class/powercap/amd-rapl/amd-rapl:%d/max_energy_range_uj", pkg);

            uint64_t test_val = 0;
            if (cpu_telem_read_u64(energy_file, &test_val) == 0) {
                snprintf(t->pkgs[t->num_packages].energy_path, sizeof(t->pkgs[t->num_packages].energy_path), "%s", energy_file);
                snprintf(t->pkgs[t->num_packages].max_range_path, sizeof(t->pkgs[t->num_packages].max_range_path), "%s", range_file);

                uint64_t max_range = 0;
                if (cpu_telem_read_u64(range_file, &max_range) == 0) {
                    t->pkgs[t->num_packages].max_range_uj = max_range;
                } else {
                    t->pkgs[t->num_packages].max_range_uj = 0;
                }
                t->num_packages++;
            }
        }
    }

    /* 3. Scan /sys/class/hwmon for amd_energy driver (AMD EPYC Linux module) */
    if (t->num_packages == 0) {
        for (int h = 0; h < 8; h++) {
            char name_path[128];
            char name_buf[64] = {0};
            snprintf(name_path, sizeof(name_path), "/sys/class/hwmon/hwmon%d/name", h);
            FILE *nf = fopen(name_path, "r");
            if (nf) {
                if (fgets(name_buf, sizeof(name_buf), nf)) {
                    if (strstr(name_buf, "amd_energy") != NULL) {
                        for (int e = 1; e <= CPU_TELEM_MAX_PACKAGES; e++) {
                            char epath[256];
                            snprintf(epath, sizeof(epath), "/sys/class/hwmon/hwmon%d/energy%d_input", h, e);
                            uint64_t test_val = 0;
                            if (cpu_telem_read_u64(epath, &test_val) == 0) {
                                snprintf(t->pkgs[t->num_packages].energy_path, sizeof(t->pkgs[t->num_packages].energy_path), "%s", epath);
                                t->pkgs[t->num_packages].max_range_path[0] = '\0';
                                t->pkgs[t->num_packages].max_range_uj = 0;
                                t->num_packages++;
                            }
                        }
                    }
                }
                fclose(nf);
            }
            if (t->num_packages > 0) break;
        }
    }

    /* 4. Direct MSR fallback (AMD EPYC MSR 0xC0010299 or Intel MSR 0x611) */
    if (t->num_packages == 0) {
        long num_cpus = sysconf(_SC_NPROCESSORS_CONF);
        if (num_cpus <= 0) num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
        if (num_cpus <= 0) num_cpus = 64;

        int seen_pkg[CPU_TELEM_MAX_PACKAGES];
        memset(seen_pkg, 0, sizeof(seen_pkg));

        for (long c = 0; c < num_cpus && t->num_packages < CPU_TELEM_MAX_PACKAGES; c++) {
            char pkg_path[128];
            snprintf(pkg_path, sizeof(pkg_path),
                     "/sys/devices/system/cpu/cpu%ld/topology/physical_package_id", c);
            uint64_t pkg_id = 0;
            if (cpu_telem_read_u64(pkg_path, &pkg_id) != 0) {
                if (c == 0) pkg_id = 0;
                else continue;
            }

            if (pkg_id >= CPU_TELEM_MAX_PACKAGES) continue;
            if (seen_pkg[pkg_id]) continue;

            uint64_t val = 0;
            uint64_t pwr_unit = 0;
            int is_amd = 0;
            int is_intel = 0;

            if (cpu_telem_read_msr((int)c, AMD_MSR_PKG_ENERGY, &val) == 0 &&
                cpu_telem_read_msr((int)c, AMD_MSR_PWR_UNIT, &pwr_unit) == 0) {
                is_amd = 1;
            } else if (cpu_telem_read_msr((int)c, INTEL_MSR_PKG_ENERGY, &val) == 0 &&
                       cpu_telem_read_msr((int)c, INTEL_MSR_PWR_UNIT, &pwr_unit) == 0) {
                is_intel = 1;
            }

            if (is_amd || is_intel) {
                seen_pkg[pkg_id] = 1;
                int idx = t->num_packages;
                t->pkgs[idx].is_msr = is_amd ? 1 : 2;
                t->pkgs[idx].msr_cpu = (int)c;

                unsigned int esu = (unsigned int)((pwr_unit >> 8) & 0x1F);
                if (esu == 0) esu = 16;
                t->pkgs[idx].msr_energy_unit_j = pow(0.5, (double)esu);

                snprintf(t->pkgs[idx].energy_path, sizeof(t->pkgs[idx].energy_path),
                         "/dev/cpu/%ld/msr (reg 0x%X)", c, is_amd ? AMD_MSR_PKG_ENERGY : INTEL_MSR_PKG_ENERGY);
                t->pkgs[idx].max_range_uj = 0;
                t->num_packages++;
            }
        }

        /* Fallback if physical_package_id was unavailable entirely */
        if (t->num_packages == 0) {
            uint64_t val = 0, pwr_unit = 0;
            if (cpu_telem_read_msr(0, AMD_MSR_PKG_ENERGY, &val) == 0 &&
                cpu_telem_read_msr(0, AMD_MSR_PWR_UNIT, &pwr_unit) == 0) {
                int idx = t->num_packages;
                t->pkgs[idx].is_msr = 1;
                t->pkgs[idx].msr_cpu = 0;
                unsigned int esu = (unsigned int)((pwr_unit >> 8) & 0x1F);
                if (esu == 0) esu = 16;
                t->pkgs[idx].msr_energy_unit_j = pow(0.5, (double)esu);
                snprintf(t->pkgs[idx].energy_path, sizeof(t->pkgs[idx].energy_path),
                         "/dev/cpu/0/msr (reg 0x%X)", AMD_MSR_PKG_ENERGY);
                t->num_packages++;
            }
        }
    }

    if (t->num_packages > 0) {
        t->rapl_available = 1;
    }

    return 0;
}

/* Start measurement: captures starting RAPL counters or MSR ticks and monotonic clock */
static inline void cpu_telemetry_start(cpu_telemetry_t *t) {
    if (!t) return;

    if (t->rapl_available) {
        for (int i = 0; i < t->num_packages; i++) {
            if (t->pkgs[i].is_msr) {
                uint32_t reg = (t->pkgs[i].is_msr == 1) ? AMD_MSR_PKG_ENERGY : INTEL_MSR_PKG_ENERGY;
                uint64_t raw = 0;
                if (cpu_telem_read_msr(t->pkgs[i].msr_cpu, reg, &raw) == 0) {
                    t->pkgs[i].msr_start_ticks = (uint32_t)(raw & 0xFFFFFFFFULL);
                }
            } else {
                uint64_t uj = 0;
                if (cpu_telem_read_u64(t->pkgs[i].energy_path, &uj) == 0) {
                    t->pkgs[i].start_uj = uj;
                }
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &t->start_time);
}

/* Stop measurement: captures ending counters, computes elapsed time & average watts */
static inline void cpu_telemetry_stop(cpu_telemetry_t *t) {
    if (!t) return;

    clock_gettime(CLOCK_MONOTONIC_RAW, &t->end_time);

    t->elapsed_sec = (double)(t->end_time.tv_sec - t->start_time.tv_sec) +
                     (double)(t->end_time.tv_nsec - t->start_time.tv_nsec) * 1e-9;
    t->elapsed_ms = t->elapsed_sec * 1000.0;

    double total_delta_uj = 0.0;

    if (t->rapl_available) {
        for (int i = 0; i < t->num_packages; i++) {
            if (t->pkgs[i].is_msr) {
                uint32_t reg = (t->pkgs[i].is_msr == 1) ? AMD_MSR_PKG_ENERGY : INTEL_MSR_PKG_ENERGY;
                uint64_t raw = 0;
                if (cpu_telem_read_msr(t->pkgs[i].msr_cpu, reg, &raw) == 0) {
                    t->pkgs[i].msr_end_ticks = (uint32_t)(raw & 0xFFFFFFFFULL);
                    uint32_t delta_ticks = t->pkgs[i].msr_end_ticks - t->pkgs[i].msr_start_ticks;
                    double delta_joules = (double)delta_ticks * t->pkgs[i].msr_energy_unit_j;
                    total_delta_uj += (delta_joules * 1e6);
                }
            } else {
                uint64_t end_uj = 0;
                if (cpu_telem_read_u64(t->pkgs[i].energy_path, &end_uj) == 0) {
                    t->pkgs[i].end_uj = end_uj;

                    /* Handle counter wrap-around */
                    if (end_uj >= t->pkgs[i].start_uj) {
                        total_delta_uj += (double)(end_uj - t->pkgs[i].start_uj);
                    } else if (t->pkgs[i].max_range_uj > 0) {
                        total_delta_uj += (double)((t->pkgs[i].max_range_uj - t->pkgs[i].start_uj) + end_uj);
                    }
                }
            }
        }
    }

    if (t->elapsed_sec > 1e-6 && total_delta_uj > 0.0) {
        /* Watts = Joules / Seconds = (microjoules / 1e6) / seconds */
        t->avg_power_watts = (total_delta_uj * 1e-6) / t->elapsed_sec;
    } else {
        t->avg_power_watts = 0.0;
    }

    t->avg_freq_mhz = cpu_telem_read_avg_freq_mhz();
    t->max_temp_c   = cpu_telem_read_max_temp_c();
}
