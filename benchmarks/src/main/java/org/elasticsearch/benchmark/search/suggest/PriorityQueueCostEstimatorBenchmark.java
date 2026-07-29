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
import org.apache.lucene.search.suggest.document.TopSuggestDocs.SuggestScoreDoc;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.PriorityQueue;
import org.elasticsearch.search.suggest.completion.CompletionSuggester;
import org.elasticsearch.search.suggest.phrase.Correction;
import org.elasticsearch.search.suggest.phrase.DirectCandidateGenerator.Candidate;
import org.elasticsearch.search.suggest.phrase.PhraseSuggester;
import org.elasticsearch.search.suggest.term.TermSuggester;
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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Checks the circuit-breaker reservation each suggester makes in {@code innerExecute} - {@code collectorReservationBytes(...)} on
 * {@link TermSuggester}, {@link PhraseSuggester} and {@link CompletionSuggester} - against the <em>real</em> heap the collector
 * queues it builds occupy.
 * <p>
 * The reservation is a cheap arithmetic formula, so comparing it to another {@code RamUsageEstimator}-based formula would be
 * circular. The independent baseline is the JVM's own allocation counter: run {@link #build} with {@code -prof gc} and read
 * {@code gc.alloc.rate.norm} - the bytes actually allocated constructing and filling the real collectors (a {@link SuggestWordQueue}
 * for term; a {@code PriorityQueue<Correction>} plus one {@link SuggestWordQueue} per generator for phrase; a
 * {@code PriorityQueue<SuggestScoreDoc>} with key <em>and</em> context text for completion). The {@code estimatedBytes} aux
 * counter is the production reservation, printed alongside; it should stay at or above {@code gc.alloc.rate.norm} (the strict
 * over-budget goal).
 * <pre>{@code
 * ../gradlew run --args "org.elasticsearch.benchmark.search.suggest.PriorityQueueCostEstimatorBenchmark.build -prof gc"
 * }</pre>
 * The {@link Scenario} params include deliberately weird sizes - {@code shard_size}/generator size of {@code 0}, {@code 1} and
 * {@link Integer#MAX_VALUE} (and {@code MAX_VALUE - 17}, the value from the incident), and generator counts up to 16. Queues that
 * large cannot be allocated (Lucene rejects {@code maxSize >= MAX_ARRAY_LENGTH} and an {@code Object[MAX_VALUE]} is ~8-16 GB), so
 * those scenarios are {@code buildable == false}: {@link #build} skips them and only {@link #estimate} runs, confirming the
 * reservation is a huge positive value (it never overflows to a negative that would bypass the breaker).
 */
@Fork(1)
@Warmup(iterations = 1)
@Measurement(iterations = 1)
@State(Scope.Benchmark)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@BenchmarkMode(Mode.AverageTime)
@SuppressWarnings("unused") // invoked by JMH
public class PriorityQueueCostEstimatorBenchmark {

    private enum Kind {
        TERM,
        PHRASE,
        COMPLETION
    }

    /**
     * A representative (or deliberately weird) {@code innerExecute} request: which suggester, the collector {@code shardSize}, the
     * phrase {@code direct_generator} count/size and token limit, and the suggested word length. {@code buildable} is false for
     * sizes too large to allocate a real queue (estimate-only).
     */
    public enum Scenario {
        // Realistic, buildable - validate reservation >= real heap via -prof gc.
        TERM_TYPICAL(Kind.TERM, 256, 0, 0, 0, 8, false, true),
        TERM_LARGE(Kind.TERM, 65536, 0, 0, 0, 8, false, true),
        PHRASE_TYPICAL(Kind.PHRASE, 256, 1, 64, 3, 8, false, true),
        PHRASE_MANY_GENERATORS(Kind.PHRASE, 256, 5, 64, 5, 8, false, true),
        PHRASE_SIXTEEN_GENERATORS(Kind.PHRASE, 1024, 16, 64, 5, 24, false, true),
        COMPLETION_TYPICAL(Kind.COMPLETION, 256, 0, 0, 0, 8, false, true),
        COMPLETION_WITH_CONTEXT(Kind.COMPLETION, 256, 0, 0, 0, 24, false, true),
        COMPLETION_SKIP_DUPLICATES(Kind.COMPLETION, 256, 0, 0, 0, 8, true, true),
        COMPLETION_LARGE(Kind.COMPLETION, 65536, 0, 0, 0, 8, false, true),
        // Edge sizes, buildable.
        SIZE_ZERO(Kind.TERM, 0, 0, 0, 0, 8, false, true),
        SIZE_ONE(Kind.TERM, 1, 0, 0, 0, 8, false, true),
        // Weird / huge - estimate-only (cannot allocate MAX_VALUE-sized queues).
        TERM_MAX(Kind.TERM, Integer.MAX_VALUE, 0, 0, 0, 8, false, false),
        TERM_MAX_MINUS_17(Kind.TERM, Integer.MAX_VALUE - 17, 0, 0, 0, 8, false, false),
        COMPLETION_MAX(Kind.COMPLETION, Integer.MAX_VALUE, 0, 0, 0, 8, false, false),
        PHRASE_MAX_SHARD(Kind.PHRASE, Integer.MAX_VALUE, 1, 64, 3, 8, false, false),
        PHRASE_MAX_GENERATORS(Kind.PHRASE, 256, 3, Integer.MAX_VALUE, 3, 8, false, false),
        PHRASE_MAX_EVERYTHING(Kind.PHRASE, Integer.MAX_VALUE, 16, Integer.MAX_VALUE, 10, 64, false, false);

        private final Kind kind;
        private final int shardSize;
        private final int numGenerators;
        private final int generatorSize;
        private final int tokenLimit;
        private final int wordLength;
        private final boolean skipDuplicates;
        private final boolean buildable;

        Scenario(
            Kind kind,
            int shardSize,
            int numGenerators,
            int generatorSize,
            int tokenLimit,
            int wordLength,
            boolean skipDuplicates,
            boolean buildable
        ) {
            this.kind = kind;
            this.shardSize = shardSize;
            this.numGenerators = numGenerators;
            this.generatorSize = generatorSize;
            this.tokenLimit = tokenLimit;
            this.wordLength = wordLength;
            this.skipDuplicates = skipDuplicates;
            this.buildable = buildable;
        }

        private int[] generatorSizes() {
            int[] sizes = new int[numGenerators];
            Arrays.fill(sizes, generatorSize);
            return sizes;
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

    /** The production circuit-breaker reservation for the scenario's suggester. */
    private long estimatedBytes() {
        return switch (scenario.kind) {
            case TERM -> TermSuggester.collectorReservationBytes(scenario.shardSize);
            case COMPLETION -> CompletionSuggester.collectorReservationBytes(scenario.shardSize, scenario.skipDuplicates);
            case PHRASE -> PhraseSuggester.collectorReservationBytes(scenario.shardSize, scenario.tokenLimit, scenario.generatorSizes());
        };
    }

    @Benchmark
    public long estimate(Metrics metrics) {
        metrics.estimatedBytes = estimatedBytes();
        return estimatedBytes();
    }

    @Benchmark
    public Object build(Metrics metrics) {
        metrics.estimatedBytes = estimatedBytes();
        if (scenario.buildable == false) {
            return null; // huge sizes cannot be allocated; those scenarios are estimate-only
        }
        // Build and fill the real collectors so -prof gc's gc.alloc.rate.norm is the real heap the reservation is validated against.
        return switch (scenario.kind) {
            case TERM -> buildSuggestWordQueue(scenario.shardSize);
            case COMPLETION -> buildCompletionQueue(scenario.shardSize);
            case PHRASE -> buildPhraseCollectors();
        };
    }

    private List<Object> buildPhraseCollectors() {
        List<Object> collectors = new ArrayList<>(scenario.numGenerators + 1);
        collectors.add(buildCorrectionQueue(scenario.shardSize, scenario.tokenLimit));
        for (int g = 0; g < scenario.numGenerators; g++) {
            collectors.add(buildSuggestWordQueue(scenario.generatorSize));
        }
        return collectors;
    }

    private SuggestWordQueue buildSuggestWordQueue(int size) {
        SuggestWordQueue queue = new SuggestWordQueue(size);
        for (int i = 0; i < size; i++) {
            SuggestWord suggestWord = new SuggestWord();
            suggestWord.string = new String(word); // fresh String + compact byte[] per element
            suggestWord.freq = i;
            suggestWord.score = i;
            queue.insertWithOverflow(suggestWord);
        }
        return queue;
    }

    private PriorityQueue<Correction> buildCorrectionQueue(int size, int candidatesPerEntry) {
        PriorityQueue<Correction> queue = new PriorityQueue<>(size) {
            @Override
            protected boolean lessThan(Correction a, Correction b) {
                return a.compareTo(b) < 0;
            }
        };
        for (int i = 0; i < size; i++) {
            Candidate[] candidates = new Candidate[candidatesPerEntry];
            for (int c = 0; c < candidates.length; c++) {
                // Copy the pre-built bytes so each BytesRef retains its own array without allocating a transient String.
                candidates[c] = new Candidate(new BytesRef(Arrays.copyOf(wordBytes, wordBytes.length)), new TermStats(1, 1L), i, i, false);
            }
            queue.insertWithOverflow(new Correction(i, candidates));
        }
        return queue;
    }

    private PriorityQueue<SuggestScoreDoc> buildCompletionQueue(int size) {
        PriorityQueue<SuggestScoreDoc> queue = new PriorityQueue<>(size) {
            @Override
            protected boolean lessThan(SuggestScoreDoc a, SuggestScoreDoc b) {
                return a.score < b.score;
            }
        };
        for (int i = 0; i < size; i++) {
            // Populate both key and context so gc.alloc.rate.norm covers the two CharSequences the reservation charges.
            queue.insertWithOverflow(new SuggestScoreDoc(i, new String(word), new String(word), i));
        }
        return queue;
    }
}
