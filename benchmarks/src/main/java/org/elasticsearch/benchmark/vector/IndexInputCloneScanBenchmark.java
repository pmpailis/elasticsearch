/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.benchmark.vector;

import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;
import org.elasticsearch.core.IOUtils;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

/**
 * Measures whether W threads concurrently scanning byte ranges of a single mmapped file through
 * per-thread {@link IndexInput#slice} handles (created on the worker thread, off one shared-arena
 * {@code MemorySegmentIndexInput}) scale as well as W threads scanning through independently opened
 * inputs (the no-sharing baseline). This validates the foundation of within-segment parallel IVF
 * search: a prior experiment attributed a large slowdown to shared-arena refcount contention, and
 * this benchmark distinguishes clone-per-worker (cold-path refcounting only) from that failure mode.
 */
@Fork(value = 1)
@Warmup(iterations = 5)
@Measurement(iterations = 5)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
public class IndexInputCloneScanBenchmark {

    private static final String FILE_NAME = "clone-scan-bench.bin";

    @Param({ "1", "2", "4", "8" })
    private int threads;

    /** shared = slices off one shared-arena input; separate = each thread opens its own input */
    @Param({ "shared", "separate" })
    private String inputMode;

    @Param({ "256" })
    private int fileMB;

    private Path tmpDir;
    private Directory dir;
    private IndexInput sharedInput;
    private List<IndexInput> separateInputs;
    private ExecutorService executor;
    private long regionLength;

    @Setup
    public void setUp() throws IOException {
        tmpDir = Files.createTempDirectory("clone-scan-bench");
        dir = new MMapDirectory(tmpDir);
        long fileLength = fileMB * 1024L * 1024L;
        try (IndexOutput out = dir.createOutput(FILE_NAME, IOContext.DEFAULT)) {
            Random random = new Random(42);
            byte[] block = new byte[1 << 20];
            for (long written = 0; written < fileLength; written += block.length) {
                random.nextBytes(block);
                out.writeBytes(block, block.length);
            }
        }
        sharedInput = dir.openInput(FILE_NAME, IOContext.DEFAULT);
        separateInputs = new ArrayList<>(threads);
        for (int i = 0; i < threads; i++) {
            separateInputs.add(dir.openInput(FILE_NAME, IOContext.DEFAULT));
        }
        executor = Executors.newFixedThreadPool(threads);
        regionLength = fileLength / threads;
    }

    @TearDown
    public void tearDown() throws IOException {
        sharedInput.close();
        for (IndexInput input : separateInputs) {
            input.close();
        }
        dir.close();
        executor.shutdownNow();
        IOUtils.rm(tmpDir);
    }

    @Benchmark
    public long scan() throws ExecutionException, InterruptedException {
        List<Future<Long>> futures = new ArrayList<>(threads);
        for (int i = 0; i < threads; i++) {
            final int worker = i;
            futures.add(executor.submit(new ScanTask(worker)));
        }
        long sum = 0;
        for (Future<Long> future : futures) {
            sum += future.get();
        }
        return sum;
    }

    private class ScanTask implements Callable<Long> {
        private final int worker;

        ScanTask(int worker) {
            this.worker = worker;
        }

        @Override
        public Long call() throws IOException {
            // The slice is created here, on the worker thread, mirroring how parallel IVF workers
            // create their posting-list slices.
            IndexInput base = "shared".equals(inputMode) ? sharedInput : separateInputs.get(worker);
            IndexInput slice = base.slice("worker-" + worker, worker * regionLength, regionLength);
            long checksum = 0;
            long longsToRead = regionLength / Long.BYTES;
            for (long i = 0; i < longsToRead; i++) {
                checksum += slice.readLong();
            }
            return checksum;
        }
    }
}
