/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.benchmark.search.suggest;

import org.apache.lucene.codecs.TermStats;
import org.apache.lucene.search.spell.SuggestWord;
import org.apache.lucene.search.spell.SuggestWordQueue;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.PriorityQueue;
import org.elasticsearch.search.suggest.Suggester;
import org.elasticsearch.search.suggest.phrase.Correction;
import org.elasticsearch.search.suggest.phrase.DirectCandidateGenerator.Candidate;
import org.elasticsearch.search.suggest.phrase.PhraseSuggester;
import org.openjdk.jmh.annotations.AuxCounters;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

import java.util.Arrays;
import java.util.concurrent.TimeUnit;

/**
 * Checks the per-entry reservation the term, phrase and completion suggesters charge on the request circuit breaker in
 * {@code innerExecute} - {@link Suggester#priorityQueueRamBytesUsed(int, long)} with a {@code SuggestWord}- or
 * {@code Correction}-shaped {@code ramBytesPerEntry} - against the <em>real</em> heap a populated queue occupies.
 * <p>
 * The estimate is a cheap arithmetic formula, so comparing it to another {@code RamUsageEstimator}-based formula would be circular.
 * The independent baseline is the JVM's own allocation counter: run {@link #build} with {@code -prof gc} and read
 * {@code gc.alloc.rate.norm} - the bytes actually allocated constructing and filling a real {@link SuggestWordQueue} (or a real
 * {@code PriorityQueue<Correction>} with {@code Candidate[]}/{@code BytesRef}/{@code TermStats} graphs). The {@code estimatedBytes}
 * aux counter is the production charge, printed alongside so the two are directly comparable; the charge should stay at or above
 * {@code gc.alloc.rate.norm} (the strict over-budget goal). For {@link #estimate}, {@code gc.alloc.rate.norm} is ~0, confirming the
 * estimate itself allocates nothing.
 * <pre>{@code
 * ../gradlew run --args "org.elasticsearch.benchmark.search.suggest.PriorityQueueCostEstimatorBenchmark.build -prof gc"
 * }</pre>
 * The {@link Scenario} params cover the {@code SuggestWord} queues (term / generators / completion, {@code candidatesPerEntry == 0})
 * and the heavier phrase {@code Correction} queue ({@code candidatesPerEntry > 0}, one {@code Candidate} per phrase token).
 */
@Fork(1)
@Warmup(iterations = 1)
@Measurement(iterations = 1)
@State(Scope.Benchmark)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@BenchmarkMode(Mode.AverageTime)
@SuppressWarnings("unused") // invoked by JMH
public class PriorityQueueCostEstimatorBenchmark {

    /**
     * Representative queue sizes ({@code shard_size} / {@code direct_generator} size), suggested word length, and candidates per
     * entry ({@code 0} = {@code SuggestWordQueue}, {@code > 0} = phrase {@code Correction} queue with that many candidates).
     */
    public enum Scenario {
        WORD_TINY(1, 8, 0),
        WORD_TYPICAL(256, 8, 0),
        WORD_LARGE(4096, 8, 0),
        WORD_LONG(4096, 64, 0),
        CORRECTION_1_TOKEN(256, 8, 1),
        CORRECTION_3_TOKEN(256, 8, 3),
        CORRECTION_5_TOKEN(4096, 8, 5);

        private final int size;
        private final int wordLength;
        private final int candidatesPerEntry;

        Scenario(int size, int wordLength, int candidatesPerEntry) {
            this.size = size;
            this.wordLength = wordLength;
            this.candidatesPerEntry = candidatesPerEntry;
        }
    }

    @Param
    public Scenario scenario;

    private char[] word;
    private byte[] wordBytes;

    @AuxCounters(AuxCounters.Type.EVENTS)
    @State(Scope.Thread)
    public static class Metrics {
        public double estimatedBytes;
    }

    @Setup(Level.Trial)
    public void setupTrial() {
        word = new char[scenario.wordLength];
        Arrays.fill(word, 'a');
        wordBytes = new byte[scenario.wordLength];
        Arrays.fill(wordBytes, (byte) 'a');
    }

    /** The production per-entry reservation for the scenario's queue type. */
    private long estimatedBytes() {
        return scenario.candidatesPerEntry == 0
            ? Suggester.priorityQueueRamBytesUsed(scenario.size, Suggester.SUGGEST_WORD_ENTRY_RAM_BYTES)
            : Suggester.priorityQueueRamBytesUsed(scenario.size, PhraseSuggester.correctionEntryRamBytes(scenario.candidatesPerEntry));
    }

    @Benchmark
    public long estimate(Metrics metrics) {
        metrics.estimatedBytes = estimatedBytes();
        return estimatedBytes();
    }

    @Benchmark
    public Object build(Metrics metrics) {
        metrics.estimatedBytes = estimatedBytes();
        // Build and fill a real queue so -prof gc's gc.alloc.rate.norm is the real heap the estimate is validated against.
        return scenario.candidatesPerEntry == 0 ? buildSuggestWordQueue() : buildCorrectionQueue();
    }

    private SuggestWordQueue buildSuggestWordQueue() {
        SuggestWordQueue queue = new SuggestWordQueue(scenario.size);
        for (int i = 0; i < scenario.size; i++) {
            SuggestWord suggestWord = new SuggestWord();
            suggestWord.string = new String(word); // fresh String + compact byte[] per element
            suggestWord.freq = i;
            suggestWord.score = i;
            queue.insertWithOverflow(suggestWord);
        }
        return queue;
    }

    private PriorityQueue<Correction> buildCorrectionQueue() {
        PriorityQueue<Correction> queue = new PriorityQueue<>(scenario.size) {
            @Override
            protected boolean lessThan(Correction a, Correction b) {
                return a.compareTo(b) < 0;
            }
        };
        for (int i = 0; i < scenario.size; i++) {
            Candidate[] candidates = new Candidate[scenario.candidatesPerEntry];
            for (int c = 0; c < candidates.length; c++) {
                // Copy the pre-built bytes so each BytesRef retains its own array without allocating a transient String.
                candidates[c] = new Candidate(new BytesRef(Arrays.copyOf(wordBytes, wordBytes.length)), new TermStats(1, 1L), i, i, false);
            }
            queue.insertWithOverflow(new Correction(i, candidates));
        }
        return queue;
    }
}
