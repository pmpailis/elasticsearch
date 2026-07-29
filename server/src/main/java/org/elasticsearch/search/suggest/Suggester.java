/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.suggest;

import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.spell.SuggestWord;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.CharsRefBuilder;
import org.apache.lucene.util.RamUsageEstimator;

import java.io.IOException;

public abstract class Suggester<T extends SuggestionSearchContext.SuggestionContext> {

    /**
     * Conservative allowance, in bytes, for the text a suggestion entry references (the compact {@code byte[]} backing a suggested
     * word, completion key/context or phrase candidate term). Sized to comfortably cover a typical/long suggestion so the
     * reservation stays at or above the real heap rather than under-charging; only pathologically long (&gt; this many bytes) terms
     * fall below it.
     */
    public static final int SUGGEST_ENTRY_TEXT_BYTES = 64;

    /**
     * Retained size of one {@link String}/{@link CharSequence}-shaped text reference: the object shell plus a
     * {@link #SUGGEST_ENTRY_TEXT_BYTES}-byte backing array. Used for {@link SuggestWord#string} and for each of a completion
     * {@code SuggestScoreDoc}'s {@code key} and {@code context}.
     */
    public static final long SUGGEST_ENTRY_TEXT_RAM_BYTES = RamUsageEstimator.shallowSizeOfInstance(String.class) + RamUsageEstimator
        .sizeOf(new byte[SUGGEST_ENTRY_TEXT_BYTES]);

    /**
     * Retained size of one {@link BytesRef}-shaped term reference: the {@code BytesRef} shell plus a
     * {@link #SUGGEST_ENTRY_TEXT_BYTES}-byte backing array. Used for phrase {@code Candidate} terms.
     */
    public static final long SUGGEST_ENTRY_BYTES_REF_RAM_BYTES = RamUsageEstimator.shallowSizeOfInstance(BytesRef.class) + RamUsageEstimator
        .sizeOf(new byte[SUGGEST_ENTRY_TEXT_BYTES]);

    /**
     * Conservative retained size of one populated {@code SuggestWordQueue} slot: a {@link SuggestWord} plus its
     * {@link #SUGGEST_ENTRY_TEXT_RAM_BYTES} text. Charged per entry by the term suggester and the phrase
     * {@code DirectCandidateGenerator} queues.
     */
    public static final long SUGGEST_WORD_ENTRY_RAM_BYTES = RamUsageEstimator.shallowSizeOfInstance(SuggestWord.class)
        + SUGGEST_ENTRY_TEXT_RAM_BYTES;

    /**
     * Conservative allowance for the {@code PriorityQueue} instance itself (its {@code size}/{@code maxSize}/{@code heap} fields plus
     * a subclass's comparator), charged once per queue on top of the backing array and the entries so the reservation stays at or
     * above the real footprint even when the per-entry charge exactly matches reality (e.g. a word at the text-allowance limit).
     */
    private static final long PRIORITY_QUEUE_SHELL_RAM_BYTES = 64;

    /** Adds two non-negative longs, saturating to {@link Long#MAX_VALUE} on overflow. */
    public static long saturatingAdd(long a, long b) {
        long sum = a + b;
        // Both inputs are non-negative, so a negative sum means the addition overflowed.
        return sum < 0 ? Long.MAX_VALUE : sum;
    }

    /**
     * Estimates the peak heap of a Lucene {@link org.apache.lucene.util.PriorityQueue} of the given size whose populated slots each
     * retain {@code ramBytesPerEntry} bytes. Shared by the term, phrase and completion suggesters, which build such queues (the
     * {@code SuggestWordQueue} of {@code DirectSpellChecker}, the phrase {@code CandidateScorer}'s {@code PriorityQueue<Correction>}
     * and the {@code SuggestScoreDocPriorityQueue} of the completion collector) sized to {@code shard_size} or a generator size.
     * <p>
     * The queue pre-allocates an {@code Object[size + 1]} backing array up-front and, once populated, holds up to {@code size}
     * entries. The reservation covers the queue shell, the backing array (which dominates and, for the pathological
     * {@code shard_size}/generator sizes this guards against, is what OOMs the node) and {@code size × ramBytesPerEntry} for the
     * entries. Each caller passes the retained size of its own entry type (e.g. {@link #SUGGEST_WORD_ENTRY_RAM_BYTES}), sized
     * conservatively so the reservation stays at or above the real peak rather than under-charging - the safe direction, since
     * over-charging at worst trips a recoverable {@code CircuitBreakingException} whereas under-charging risks an OOM. The
     * multiplication is saturated to {@link Long#MAX_VALUE} on overflow, which trips any real breaker.
     * <p>
     * Validated against the JVM's own allocation counter (JMH {@code -prof gc}, {@code gc.alloc.rate.norm}) via
     * {@code PriorityQueueCostEstimatorBenchmark}: populated term/phrase/completion collectors should stay at or below the
     * corresponding {@code collectorReservationBytes} charge.
     */
    public static long priorityQueueRamBytesUsed(int size, long ramBytesPerEntry) {
        long backingArray = RamUsageEstimator.alignObjectSize(
            (long) RamUsageEstimator.NUM_BYTES_ARRAY_HEADER + (size + 1L) * RamUsageEstimator.NUM_BYTES_OBJECT_REF
        );
        long entries;
        try {
            entries = Math.multiplyExact((long) size, ramBytesPerEntry);
        } catch (ArithmeticException overflow) {
            return Long.MAX_VALUE;
        }
        long total = PRIORITY_QUEUE_SHELL_RAM_BYTES + backingArray + entries;
        return total < 0 ? Long.MAX_VALUE : total;
    }

    protected abstract Suggest.Suggestion<? extends Suggest.Suggestion.Entry<? extends Suggest.Suggestion.Entry.Option>> innerExecute(
        String name,
        T suggestion,
        IndexSearcher searcher,
        CharsRefBuilder spare
    ) throws IOException;

    protected abstract Suggest.Suggestion<? extends Suggest.Suggestion.Entry<? extends Suggest.Suggestion.Entry.Option>> emptySuggestion(
        String name,
        T suggestion,
        CharsRefBuilder spare
    ) throws IOException;

    public Suggest.Suggestion<? extends Suggest.Suggestion.Entry<? extends Suggest.Suggestion.Entry.Option>> execute(
        String name,
        T suggestion,
        IndexSearcher searcher,
        CharsRefBuilder spare
    ) throws IOException {

        // we only want to output an empty suggestion on empty shards
        if (searcher.getIndexReader().numDocs() == 0) {
            return emptySuggestion(name, suggestion, spare);
        }
        return innerExecute(name, suggestion, searcher, spare);
    }

}
