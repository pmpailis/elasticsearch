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
import org.apache.lucene.util.CharsRefBuilder;
import org.apache.lucene.util.RamUsageEstimator;

import java.io.IOException;

public abstract class Suggester<T extends SuggestionSearchContext.SuggestionContext> {

    /**
     * Conservative retained size of a single populated queue slot: the entry object plus the {@code String} wrapper it references.
     * Modelled on {@link SuggestWord} - the entry of the term/phrase generator {@code SuggestWordQueue} - as a representative for
     * the other small suggestion entries (the phrase {@code Correction} queue and the completion {@code SuggestScoreDoc} queue).
     * Only the fixed per-entry objects are counted; the {@code String}'s variable-length backing bytes are a bounded residual left
     * uncharged, which is acceptable because it can only be reached once the queue is actually populated with that many entries.
     */
    private static final long PRIORITY_QUEUE_ENTRY_RAM_BYTES = RamUsageEstimator.shallowSizeOfInstance(SuggestWord.class)
        + RamUsageEstimator.shallowSizeOfInstance(String.class);

    /**
     * Estimates the peak heap of a Lucene {@link org.apache.lucene.util.PriorityQueue} of the given size. Shared by the term,
     * phrase and completion suggesters, which build such queues (the {@code SuggestWordQueue} of {@code DirectSpellChecker} and the
     * {@code SuggestScoreDocPriorityQueue} of the completion collector) sized to {@code shard_size} or a generator size.
     * <p>
     * The queue pre-allocates an {@code Object[size + 1]} backing array up-front and, once populated, holds up to {@code size}
     * entries. The reservation covers <em>both</em>: the backing array plus {@code size} entries of
     * {@link #PRIORITY_QUEUE_ENTRY_RAM_BYTES} each, so it no longer under-charges the breaker by ignoring the elements the queue
     * holds. The multiplication is saturated to {@link Long#MAX_VALUE} on overflow, which trips any real breaker.
     */
    public static long priorityQueueRamBytesUsed(int size) {
        long backingArray = RamUsageEstimator.alignObjectSize(
            (long) RamUsageEstimator.NUM_BYTES_ARRAY_HEADER + (size + 1L) * RamUsageEstimator.NUM_BYTES_OBJECT_REF
        );
        long entries;
        try {
            entries = Math.multiplyExact(size, PRIORITY_QUEUE_ENTRY_RAM_BYTES);
        } catch (ArithmeticException ex) {
            return Long.MAX_VALUE;
        }
        long total = backingArray + entries;
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
