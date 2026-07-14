/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */
package org.elasticsearch.search.vectors;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.search.join.CheckJoinIndex;
import org.apache.lucene.search.join.QueryBitSetProducer;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.BitSet;
import org.elasticsearch.common.logging.LogConfigurator;
import org.elasticsearch.index.codec.vectors.diskbbq.ES920DiskBBQVectorsFormat;
import org.elasticsearch.index.codec.vectors.diskbbq.TestIvfQueryConfigResolver;
import org.elasticsearch.index.codec.vectors.diskbbq.next.ESNextDiskBBQVectorsFormat;
import org.junit.After;
import org.junit.Before;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static com.carrotsearch.randomizedtesting.RandomizedTest.randomBoolean;
import static com.carrotsearch.randomizedtesting.RandomizedTest.randomFloat;
import static com.carrotsearch.randomizedtesting.RandomizedTest.randomInt;
import static com.carrotsearch.randomizedtesting.RandomizedTest.randomIntBetween;

/**
 * Verifies query-level cross-segment best-first scheduling. At a visit ratio of 1.0 every posting of every leaf is
 * visited under both regimes, so results must be identical to the scheduler-off baseline (modulo exactly-tied
 * scores) — this carries the correctness weight: registration, best-first claiming, worker merges, per-leaf
 * continuations and the final collection all round-trip through it. Partial budgets change the visit
 * <i>distribution</i> by design (that is the point of the scheduler), so partial-ratio checks assert the provable
 * relations only. The scheduling flag and worker config are injected via query subclasses because the sysprops are
 * static finals, unreliable in a shared test JVM.
 */
public class IVFCrossSegmentSchedulerTests extends LuceneTestCase {

    static {
        LogConfigurator.configureESLogging(); // native access requires logging to be initialized
    }

    private static final String FIELD = "field";
    private static final String FILTER_FIELD = "_filter";
    private static final String FILTER_MATCH = "match";

    private ExecutorService executor;
    private TaskExecutor taskExecutor;
    private int schedulerWorkers;

    @Before
    public void setUpExecutor() {
        schedulerWorkers = randomIntBetween(2, 4);
        executor = Executors.newFixedThreadPool(schedulerWorkers);
        taskExecutor = new TaskExecutor(executor);
    }

    @After
    public void tearDownExecutor() {
        executor.shutdownNow();
    }

    public void testExhaustiveParityMultiSegment() throws IOException {
        int dims = randomIntBetween(16, 48);
        int segments = randomIntBetween(2, 4);
        int perSegment = randomIntBetween(1000, 1500);
        try (Directory dir = newDirectory()) {
            indexSegments(dir, segments, perSegment, dims, 0);
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                assertEquals(segments, reader.leaves().size());
                IndexSearcher searcher = new IndexSearcher(reader);

                // reachability probe: a cancelling config must abort the scan from a worker chunk, proving the
                // scheduler actually claims work (the parity assertions below would pass vacuously otherwise)
                Query probe = schedulerQuery(randomVector(dims), 10, null, 1.0f, () -> { throw new RuntimeException("cancelled"); });
                RuntimeException e = expectThrows(RuntimeException.class, () -> searcher.search(probe, 10));
                assertCancelled(e);

                for (int i = 0; i < 5; i++) {
                    float[] query = randomVector(dims);
                    int k = randomIntBetween(5, 20);
                    TopDocs baseline = searcher.search(baselineQuery(query, k, null, 1.0f), k);
                    TopDocs scheduled = searcher.search(schedulerQuery(query, k, null, 1.0f, null), k);
                    IVFIntraSegmentParallelScanTests.assertSameTopDocs(baseline, scheduled);

                    // partial budgets redistribute visits by design; the provable relation is against the full run
                    float visitRatio = randomFloat() * 0.15f + 0.05f;
                    TopDocs partial = searcher.search(schedulerQuery(query, k, null, visitRatio, null), k);
                    assertEquals(k, partial.scoreDocs.length);
                    float fullKth = scheduled.scoreDocs[scheduled.scoreDocs.length - 1].score;
                    float partialKth = partial.scoreDocs[partial.scoreDocs.length - 1].score;
                    assertTrue(
                        "full-budget k-th score " + fullKth + " must be >= partial-budget k-th score " + partialKth,
                        fullKth >= partialKth
                    );
                }
            }
        }
    }

    public void testFilteredAndDeletesParity() throws IOException {
        int dims = randomIntBetween(16, 48);
        int segments = randomIntBetween(2, 3);
        int perSegment = randomIntBetween(1000, 1500);
        try (Directory dir = newDirectory()) {
            indexSegments(dir, segments, perSegment, dims, perSegment / 5);
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                assertNotNull("test expects deletes to produce live docs", reader.leaves().get(0).reader().getLiveDocs());
                IndexSearcher searcher = new IndexSearcher(reader);
                for (int i = 0; i < 5; i++) {
                    float[] query = randomVector(dims);
                    int k = randomIntBetween(5, 20);
                    Query filter = randomBoolean() ? new TermQuery(new Term(FILTER_FIELD, FILTER_MATCH)) : null;
                    TopDocs baseline = searcher.search(baselineQuery(query, k, filter, 1.0f), k);
                    TopDocs scheduled = searcher.search(schedulerQuery(query, k, filter, 1.0f, null), k);
                    IVFIntraSegmentParallelScanTests.assertSameTopDocs(baseline, scheduled);
                }
            }
        }
    }

    public void testDiversifiedParity() throws IOException {
        int dims = randomIntBetween(16, 32);
        int segments = 2;
        int parentsPerSegment = randomIntBetween(120, 180);
        try (Directory dir = newDirectory()) {
            indexParentChildSegments(dir, segments, parentsPerSegment, dims);
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                IndexSearcher searcher = new IndexSearcher(reader);
                BitSetProducer parentsFilter = new QueryBitSetProducer(new TermQuery(new Term("docType", "_parent")));
                CheckJoinIndex.check(reader, parentsFilter);
                for (int i = 0; i < 5; i++) {
                    float[] query = randomVector(dims);
                    int k = randomIntBetween(5, 15);
                    TopDocs baseline = searcher.search(diversifiedQuery(query, k, parentsFilter, false), k);
                    TopDocs scheduled = searcher.search(diversifiedQuery(query, k, parentsFilter, true), k);
                    IVFIntraSegmentParallelScanTests.assertSameTopDocs(baseline, scheduled);
                    assertDistinctParents(scheduled, reader, parentsFilter);
                }
            }
        }
    }

    /** ES920 leaves have no parallel-scan support: they must complete serially inside their leaf task (hybrid mode). */
    public void testMixedEligibilityParity() throws IOException {
        int dims = randomIntBetween(16, 48);
        int perSegment = randomIntBetween(1000, 1500);
        try (Directory dir = newDirectory()) {
            writeSegment(dir, esNextFormat(), perSegment, dims);
            writeSegment(dir, new ES920DiskBBQVectorsFormat(64, 16), perSegment, dims);
            writeSegment(dir, esNextFormat(), perSegment, dims);
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                assertEquals(3, reader.leaves().size());
                IndexSearcher searcher = new IndexSearcher(reader);
                for (int i = 0; i < 5; i++) {
                    float[] query = randomVector(dims);
                    int k = randomIntBetween(5, 20);
                    TopDocs baseline = searcher.search(baselineQuery(query, k, null, 1.0f), k);
                    TopDocs scheduled = searcher.search(schedulerQuery(query, k, null, 1.0f, null), k);
                    IVFIntraSegmentParallelScanTests.assertSameTopDocs(baseline, scheduled);
                }
            }
        }
    }

    /** A single eligible leaf still round-trips through the scheduler correctly (register → scan → continue). */
    public void testSingleEligibleLeafParity() throws IOException {
        int dims = randomIntBetween(16, 48);
        int numVectors = randomIntBetween(2000, 3000);
        try (Directory dir = newDirectory()) {
            // two leaves so the scheduler activates, but only the ESNext one is eligible to register
            writeSegment(dir, esNextFormat(), numVectors, dims);
            writeSegment(dir, new ES920DiskBBQVectorsFormat(64, 16), numVectors / 4, dims);
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                IndexSearcher searcher = new IndexSearcher(reader);
                for (int i = 0; i < 3; i++) {
                    float[] query = randomVector(dims);
                    int k = randomIntBetween(5, 20);
                    TopDocs baseline = searcher.search(baselineQuery(query, k, null, 1.0f), k);
                    TopDocs scheduled = searcher.search(schedulerQuery(query, k, null, 1.0f, null), k);
                    IVFIntraSegmentParallelScanTests.assertSameTopDocs(baseline, scheduled);
                }
            }
        }
    }

    private IVFKnnFloatVectorQuery baselineQuery(float[] vector, int k, Query filter, float visitRatio) {
        return new IVFKnnFloatVectorQuery(FIELD, vector, k, k, filter, visitRatio, resolver()) {
            @Override
            boolean crossSegmentScheduling() {
                return false;
            }

            @Override
            IVFParallelismConfig intraSegmentParallelism() {
                return null;
            }
        };
    }

    private IVFKnnFloatVectorQuery schedulerQuery(float[] vector, int k, Query filter, float visitRatio, Runnable checkCancelled) {
        return new IVFKnnFloatVectorQuery(FIELD, vector, k, k, filter, visitRatio, resolver()) {
            @Override
            boolean crossSegmentScheduling() {
                return true;
            }

            @Override
            IVFParallelismConfig intraSegmentParallelism() {
                // maxWorkers 1: no per-leaf forking, the scheduler's workers do all parallel scanning
                return new IVFParallelismConfig(taskExecutor, 1, schedulerWorkers, checkCancelled == null ? () -> {} : checkCancelled);
            }
        };
    }

    private DiversifyingChildrenIVFKnnFloatVectorQuery diversifiedQuery(float[] vector, int k, BitSetProducer parents, boolean scheduled) {
        return new DiversifyingChildrenIVFKnnFloatVectorQuery(FIELD, vector, k, k, null, parents, 1.0f, resolver()) {
            @Override
            boolean crossSegmentScheduling() {
                return scheduled;
            }

            @Override
            IVFParallelismConfig intraSegmentParallelism() {
                return scheduled ? new IVFParallelismConfig(taskExecutor, 1, schedulerWorkers, () -> {}) : null;
            }
        };
    }

    private static TestIvfQueryConfigResolver resolver() {
        return new TestIvfQueryConfigResolver(
            ESNextDiskBBQVectorsFormat.CentroidIndexFormat.FLAT,
            ESNextDiskBBQVectorsFormat.QuantEncoding.ONE_BIT_4BIT_QUERY,
            false,
            1.0f
        );
    }

    private static KnnVectorsFormat esNextFormat() {
        return new ESNextDiskBBQVectorsFormat(64, 16, null);
    }

    private static void assertCancelled(RuntimeException e) {
        for (Throwable t = e; t != null; t = t.getCause() == t ? null : t.getCause()) {
            if (t.getMessage() != null && t.getMessage().contains("cancelled")) {
                return;
            }
        }
        throw new AssertionError("expected a cancellation failure", e);
    }

    private static void assertDistinctParents(TopDocs topDocs, DirectoryReader reader, BitSetProducer parentsFilter) throws IOException {
        Set<Long> parents = new HashSet<>();
        for (var scoreDoc : topDocs.scoreDocs) {
            int leafOrd = org.apache.lucene.index.ReaderUtil.subIndex(scoreDoc.doc, reader.leaves());
            var leaf = reader.leaves().get(leafOrd);
            BitSet parentBitSet = parentsFilter.getBitSet(leaf);
            int leafDoc = scoreDoc.doc - leaf.docBase;
            assertFalse("a returned doc must be a child", parentBitSet.get(leafDoc));
            long globalParent = ((long) leafOrd << 32) | parentBitSet.nextSetBit(leafDoc);
            assertTrue("each returned child must map to a distinct parent", parents.add(globalParent));
        }
    }

    private static void indexSegments(Directory dir, int segments, int perSegment, int dims, int deletesPerSegment) throws IOException {
        IndexWriterConfig config = new IndexWriterConfig().setCodec(TestUtil.alwaysKnnVectorsFormat(esNextFormat()))
            .setMaxBufferedDocs(perSegment * 2)
            .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH)
            .setMergePolicy(NoMergePolicy.INSTANCE);
        try (IndexWriter writer = new IndexWriter(dir, config)) {
            int id = 0;
            for (int segment = 0; segment < segments; segment++) {
                int firstId = id;
                for (int i = 0; i < perSegment; i++, id++) {
                    Document doc = new Document();
                    doc.add(new KnnFloatVectorField(FIELD, randomVector(dims), VectorSimilarityFunction.EUCLIDEAN));
                    doc.add(new StringField(FILTER_FIELD, randomBoolean() ? FILTER_MATCH : "miss", Field.Store.NO));
                    doc.add(new StringField("id", "id" + id, Field.Store.NO));
                    writer.addDocument(doc);
                }
                writer.commit();
                for (int i = 0; i < deletesPerSegment; i++) {
                    writer.deleteDocuments(new Term("id", "id" + (firstId + randomInt(perSegment - 1))));
                }
                if (deletesPerSegment > 0) {
                    writer.commit();
                }
            }
        }
    }

    private static void writeSegment(Directory dir, KnnVectorsFormat format, int numVectors, int dims) throws IOException {
        IndexWriterConfig config = new IndexWriterConfig().setCodec(TestUtil.alwaysKnnVectorsFormat(format))
            .setMaxBufferedDocs(numVectors * 2)
            .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH)
            .setMergePolicy(NoMergePolicy.INSTANCE);
        try (IndexWriter writer = new IndexWriter(dir, config)) {
            for (int i = 0; i < numVectors; i++) {
                Document doc = new Document();
                doc.add(new KnnFloatVectorField(FIELD, randomVector(dims), VectorSimilarityFunction.EUCLIDEAN));
                writer.addDocument(doc);
            }
            writer.commit();
        }
    }

    private static void indexParentChildSegments(Directory dir, int segments, int parentsPerSegment, int dims) throws IOException {
        IndexWriterConfig config = new IndexWriterConfig().setCodec(TestUtil.alwaysKnnVectorsFormat(esNextFormat()))
            .setMaxBufferedDocs(parentsPerSegment * 20)
            .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH)
            .setMergePolicy(NoMergePolicy.INSTANCE);
        try (IndexWriter writer = new IndexWriter(dir, config)) {
            for (int segment = 0; segment < segments; segment++) {
                for (int parent = 0; parent < parentsPerSegment; parent++) {
                    List<Document> block = new ArrayList<>();
                    int children = randomIntBetween(1, 8);
                    for (int child = 0; child < children; child++) {
                        Document childDoc = new Document();
                        childDoc.add(new KnnFloatVectorField(FIELD, randomVector(dims), VectorSimilarityFunction.EUCLIDEAN));
                        block.add(childDoc);
                    }
                    Document parentDoc = new Document();
                    parentDoc.add(new StringField("docType", "_parent", Field.Store.NO));
                    block.add(parentDoc);
                    writer.addDocuments(block);
                }
                writer.commit();
            }
        }
    }

    private static float[] randomVector(int dims) {
        float[] vector = new float[dims];
        for (int i = 0; i < dims; i++) {
            vector[i] = randomFloat() * 2 - 1;
        }
        return vector;
    }
}
