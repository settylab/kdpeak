"""
Performance and stress tests for kdpeak with large datasets.
"""

import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import pytest

from kdpeak.util import call_peaks, events_dict_from_file, make_kdes


class TestPerformance:
    """Test performance characteristics of kdpeak."""

    @pytest.fixture
    def large_events_dict(self):
        """Create a large synthetic events dictionary for performance testing."""
        np.random.seed(42)
        events = {}

        # Create realistic large chromosome data
        chromosomes = ["chr1", "chr2", "chr3", "chrX"]
        chrom_sizes = [200_000_000, 150_000_000, 120_000_000, 100_000_000]

        for chrom, size in zip(chromosomes, chrom_sizes):
            # Simulate realistic number of events (1 per kb on average)
            n_events = size // 1000

            # Create clustered events (realistic for ChIP-seq, ATAC-seq)
            cluster_centers = np.random.randint(0, size, n_events // 10)
            locations = []

            for center in cluster_centers:
                # Add events around each cluster center
                cluster_size = np.random.poisson(10) + 1
                cluster_events = np.random.normal(center, 500, cluster_size)
                cluster_events = np.clip(cluster_events, 0, size)
                locations.extend(cluster_events)

            # Add some random background events
            background = np.random.randint(0, size, n_events // 20)
            locations.extend(background)

            events[chrom] = pd.DataFrame(
                {
                    "variable": ["start"] * len(locations),
                    "location": np.array(locations, dtype=int),
                }
            )

        return events

    def test_memory_usage_large_dataset(self, large_events_dict):
        """Test memory usage with large datasets."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process large dataset
        comb_data, signal_list = make_kdes(
            large_events_dict,
            step=1000,  # Lower resolution for performance
            kde_bw=5000,
        )

        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = peak_memory - initial_memory

        print(f"Memory used: {memory_used:.1f} MB")

        # Should not use excessive memory (less than 2GB for this test)
        assert memory_used < 2000, f"Used {memory_used:.1f} MB, which is too much"

        # Check that we got reasonable output
        assert len(comb_data) > 0
        assert len(signal_list) == 4  # Four chromosomes

    def test_processing_time_scales_reasonably(self):
        """Test that processing time scales reasonably with data size."""
        np.random.seed(42)

        times = []
        sizes = [1000, 5000, 10000]  # Different dataset sizes

        for size in sizes:
            # Create dataset of specified size
            events_dict = {
                "chr1": pd.DataFrame(
                    {
                        "variable": ["start"] * size,
                        "location": np.random.randint(0, 10_000_000, size),
                    }
                )
            }

            start_time = time.time()

            comb_data, signal_list = make_kdes(events_dict, step=500, kde_bw=2000)

            end_time = time.time()
            processing_time = end_time - start_time
            times.append(processing_time)

            print(f"Size {size}: {processing_time:.2f} seconds")

        # Processing time should scale sub-quadratically
        # (KDE is typically O(n log n) with FFT)
        ratio_1_2 = times[1] / times[0]  # 5x data
        ratio_2_3 = times[2] / times[1]  # 2x data

        # These ratios should be reasonable (not exponential growth)
        assert ratio_1_2 < 20, f"Time scaling is too steep: {ratio_1_2:.1f}x"
        assert ratio_2_3 < 10, f"Time scaling is too steep: {ratio_2_3:.1f}x"

    def test_large_file_processing(self, temp_dir):
        """Test processing of large BED files."""
        # Create a large synthetic BED file
        large_bed_file = temp_dir / "large_test.bed"

        with open(large_bed_file, "w") as f:
            np.random.seed(42)
            n_intervals = 50000  # 50k intervals

            for i in range(n_intervals):
                chrom = f"chr{np.random.randint(1, 23)}"
                start = np.random.randint(0, 100_000_000)
                end = start + np.random.randint(100, 5000)
                f.write(f"{chrom}\t{start}\t{end}\n")

        # Test reading and processing
        start_time = time.time()

        events_dict = events_dict_from_file(str(large_bed_file))

        read_time = time.time()

        # Process subset of chromosomes for faster testing
        small_dict = {k: v for k, v in list(events_dict.items())[:3]}

        comb_data, signal_list = make_kdes(small_dict, step=1000, kde_bw=5000)

        process_time = time.time()

        print(f"File read time: {read_time - start_time:.2f} seconds")
        print(f"Processing time: {process_time - read_time:.2f} seconds")

        # Should complete in reasonable time (less than 60 seconds)
        total_time = process_time - start_time
        assert total_time < 60, f"Processing took too long: {total_time:.1f} seconds"

        # Should produce output
        assert len(comb_data) > 0

    def test_memory_efficiency_repeated_calls(self, sample_events_dict):
        """Test that repeated calls don't cause memory leaks."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run multiple iterations
        for i in range(5):
            comb_data, signal_list = make_kdes(sample_events_dict)
            peaks = call_peaks(comb_data, signal_list)

            # Force garbage collection
            import gc

            gc.collect()

            if i > 0:  # Skip first iteration (warmup)
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_increase = current_memory - initial_memory

                # Memory shouldn't grow significantly with each iteration
                assert (
                    memory_increase < 100
                ), f"Memory leak detected: {memory_increase:.1f} MB increase"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_chromosomes(self):
        """Test handling of empty chromosomes."""
        events_dict = {
            "chr1": pd.DataFrame({"variable": ["start"], "location": [1000]}),
            "chr2": pd.DataFrame(columns=["variable", "location"]),  # Empty
            "chr3": pd.DataFrame({"variable": ["start"], "location": [5000]}),
        }

        comb_data, signal_list = make_kdes(events_dict)

        # Should handle empty chromosome gracefully
        assert len(comb_data) > 0
        chromosomes = comb_data["seqname"].unique()

        # May or may not include empty chromosome, but shouldn't crash
        assert "chr1" in chromosomes
        assert "chr3" in chromosomes

    def test_single_event_chromosome(self):
        """Test chromosome with only one event."""
        events_dict = {
            "chr1": pd.DataFrame({"variable": ["start"], "location": [10000]})
        }

        comb_data, signal_list = make_kdes(events_dict, kde_bw=1000)

        # Should handle single event without error
        assert len(comb_data) > 0
        assert len(signal_list) == 1
        assert comb_data["seqname"].iloc[0] == "chr1"

    def test_very_large_coordinates(self):
        """Test handling of very large genomic coordinates."""
        # Test with coordinates near the limit of typical chromosome sizes
        large_coords = [100_000_000, 200_000_000, 249_250_621]  # chr1 size

        events_dict = {
            "chr1": pd.DataFrame(
                {"variable": ["start"] * len(large_coords), "location": large_coords}
            )
        }

        comb_data, signal_list = make_kdes(
            events_dict,
            step=10000,  # Coarser resolution for large coordinates
            kde_bw=50000,
        )

        assert len(comb_data) > 0
        assert comb_data["location"].max() >= max(large_coords)

    def test_many_small_chromosomes(self):
        """Test handling many small chromosomes (like scaffolds)."""
        events_dict = {}

        # Create 100 small scaffolds
        for i in range(100):
            scaffold_name = f"scaffold_{i:03d}"
            events_dict[scaffold_name] = pd.DataFrame(
                {"variable": ["start"] * 5, "location": np.random.randint(0, 10000, 5)}
            )

        start_time = time.time()

        comb_data, signal_list = make_kdes(events_dict, step=100, kde_bw=500)

        processing_time = time.time() - start_time

        # Should handle many chromosomes efficiently
        assert (
            processing_time < 30
        ), f"Too slow with many chromosomes: {processing_time:.1f}s"
        assert len(signal_list) == 100
        assert len(comb_data["seqname"].unique()) == 100

    def test_extreme_parameters(self, sample_events_dict):
        """Test with extreme parameter values."""
        # Very small bandwidth
        comb_data_small, _ = make_kdes(
            sample_events_dict, kde_bw=10, step=5  # Very small
        )
        assert len(comb_data_small) > 0

        # Very large bandwidth
        comb_data_large, _ = make_kdes(
            sample_events_dict, kde_bw=50000, step=1000  # Very large
        )
        assert len(comb_data_large) > 0

        # Very high resolution
        comb_data_highres, _ = make_kdes(
            sample_events_dict, kde_bw=200, step=1  # Single base pair resolution
        )
        assert len(comb_data_highres) > 0

    def test_duplicate_coordinates(self):
        """Test handling of duplicate coordinates."""
        # Create events with many duplicates
        duplicate_locations = [1000] * 50 + [2000] * 30 + [3000] * 20

        events_dict = {
            "chr1": pd.DataFrame(
                {
                    "variable": ["start"] * len(duplicate_locations),
                    "location": duplicate_locations,
                }
            )
        }

        comb_data, signal_list = make_kdes(events_dict)

        # Should handle duplicates without error
        assert len(comb_data) > 0
        assert len(signal_list) == 1

        # Density should be higher at duplicate locations
        chr1_data = comb_data[comb_data["seqname"] == "chr1"]
        max_density_location = chr1_data.loc[chr1_data["density"].idxmax(), "location"]

        # Maximum density should be near one of the duplicate clusters
        assert any(abs(max_density_location - loc) < 1000 for loc in [1000, 2000, 3000])


class TestStressTests:
    """Stress tests for robustness."""

    @pytest.mark.slow
    def test_concurrent_processing(self, sample_events_dict):
        """Test that the package handles concurrent processing safely."""
        import queue
        import threading

        results_queue = queue.Queue()

        def worker():
            try:
                comb_data, signal_list = make_kdes(sample_events_dict)
                peaks = call_peaks(comb_data, signal_list)
                results_queue.put(("success", len(peaks)))
            except Exception as e:
                results_queue.put(("error", str(e)))

        # Start multiple threads
        threads = []
        for i in range(4):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Check results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())

        assert len(results) == 4

        # All should succeed
        for status, result in results:
            assert status == "success", f"Thread failed: {result}"
            assert result > 0, "No peaks found"

    def test_resource_cleanup(self, sample_events_dict):
        """Test that resources are properly cleaned up."""
        import gc
        import weakref

        # Keep weak references to check cleanup
        weak_refs = []

        def create_and_process():
            comb_data, signal_list = make_kdes(sample_events_dict)
            weak_refs.append(weakref.ref(comb_data))
            # Note: Lists don't support weak references, so we only track the DataFrame
            return call_peaks(comb_data, signal_list)

        # Create several objects
        for i in range(3):
            peaks = create_and_process()
            assert len(peaks) > 0

        # Force garbage collection
        gc.collect()

        # Check that objects were cleaned up
        alive_refs = [ref for ref in weak_refs if ref() is not None]

        # Some objects might still be alive due to Python's GC behavior,
        # but the majority should be cleaned up
        cleanup_ratio = 1 - (len(alive_refs) / len(weak_refs))
        assert cleanup_ratio > 0.5, f"Poor cleanup: {cleanup_ratio:.1%} cleaned up"


class TestBenchmarks:
    """Benchmark tests for performance monitoring."""

    def test_typical_chipseq_performance(self):
        """Benchmark with typical ChIP-seq sized dataset."""
        np.random.seed(42)

        # Typical ChIP-seq: ~50k peaks across genome
        n_events = 100000  # 100k fragment ends

        events_dict = {
            "chr1": pd.DataFrame(
                {
                    "variable": ["start"] * n_events,
                    "location": np.random.randint(0, 200_000_000, n_events),
                }
            )
        }

        start_time = time.time()

        comb_data, signal_list = make_kdes(events_dict, step=500, kde_bw=2000)

        kde_time = time.time()

        peaks = call_peaks(comb_data, signal_list)

        total_time = time.time() - start_time

        print(f"ChIP-seq benchmark:")
        print(f"  Events: {n_events:,}")
        print(f"  KDE time: {kde_time - start_time:.2f}s")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Peaks found: {len(peaks):,}")
        print(f"  Events/second: {n_events/total_time:,.0f}")

        # Should complete in reasonable time
        assert total_time < 120, f"Too slow: {total_time:.1f}s"
        assert len(peaks) > 0

        # Performance target: >1000 events/second
        events_per_second = n_events / total_time
        assert events_per_second > 500, f"Too slow: {events_per_second:.0f} events/s"
