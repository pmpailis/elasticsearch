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
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.TestUtil;
import org.elasticsearch.common.logging.LogConfigurator;
import org.elasticsearch.index.codec.vectors.diskbbq.ES920DiskBBQVectorsFormat;
import org.elasticsearch.index.codec.vectors.diskbbq.TestIvfQueryConfigResolver;
import org.elasticsearch.index.codec.vectors.diskbbq.next.ESNextDiskBBQVectorsFormat;
import org.elasticsearch.index.mapper.vectors.DenseVectorFieldMapper;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import static com.carrotsearch.randomizedtesting.RandomizedTest.randomFloat;
import static com.carrotsearch.randomizedtesting.RandomizedTest.randomIntBetween;

/**
 * Regression test for the mixed-precondition bug: segments written with preconditioning store their IVF structures
 * over rotated vectors while other segments (older formats, or written without preconditioning) do not, so each leaf
 * must be searched with the query view matching its own space. The query used to rotate its shared {@code query}
 * field once — first preconditioning leaf wins — after which every other leaf (and the raw-vector rescore path, since
 * raw float storage is never rotated) silently searched a rotated query against unrotated data.
 */
public class IVFMixedPreconditionTests extends LuceneTestCase {

    static {
        LogConfigurator.configureESLogging(); // native access requires logging to be initialized
    }

    private static final String FIELD = "field";

    public void testMixedPreconditionShardSearchesEachLeafInItsOwnSpace() throws IOException {
        int dims = randomIntBetween(16, 48);
        int perSegment = randomIntBetween(1200, 1800);

        try (Directory mixed = newDirectory()) {
            // the mixed "shard": segment A written preconditioned (ESNext), segment B without (ES920)
            writeSegment(mixed, preconditionedFormat(), randomVectors(perSegment, dims));
            writeSegment(mixed, es920Format(), randomVectors(perSegment, dims));

            try (DirectoryReader reader = DirectoryReader.open(mixed)) {
                assertEquals(2, reader.leaves().size());
                for (int iter = 0; iter < 3; iter++) {
                    float[] queryVector = randomVector(dims);
                    float[] original = queryVector.clone();
                    int k = randomIntBetween(5, 20);

                    // per-leaf baselines over the SAME physical segments: each leaf searched alone resolves its own
                    // query view (rotated for the preconditioned leaf, original for the ES920 one)
                    List<ScoreDoc> merged = new ArrayList<>();
                    for (LeafReaderContext leaf : reader.leaves()) {
                        TopDocs leafExpected = new IndexSearcher(leaf.reader()).search(newQuery(queryVector, k), k);
                        for (ScoreDoc scoreDoc : leafExpected.scoreDocs) {
                            merged.add(new ScoreDoc(scoreDoc.doc + leaf.docBase, scoreDoc.score));
                        }
                    }
                    TopDocs expected = topK(merged, k);

                    IVFKnnFloatVectorQuery mixedQuery = newQuery(queryVector, k);
                    TopDocs actual = new IndexSearcher(reader).search(mixedQuery, k);

                    // the shared query vector must never be mutated by a search
                    assertArrayEquals(original, mixedQuery.getQuery(), 0.0f);
                    // each leaf must contribute exactly what it contributes when searched alone
                    IVFIntraSegmentParallelScanTests.assertSameTopDocs(expected, actual);
                }
            }
        }
    }

    private static IVFKnnFloatVectorQuery newQuery(float[] queryVector, int k) {
        return new IVFKnnFloatVectorQuery(
            FIELD,
            queryVector,
            k,
            k,
            null,
            1.0f,
            new TestIvfQueryConfigResolver(
                ESNextDiskBBQVectorsFormat.CentroidIndexFormat.FLAT,
                ESNextDiskBBQVectorsFormat.QuantEncoding.ONE_BIT_4BIT_QUERY,
                true,
                1.0f
            )
        );
    }

    /** Global top-k over the per-leaf baseline hits (already offset to global doc ids). */
    private static TopDocs topK(List<ScoreDoc> merged, int k) {
        merged.sort(Comparator.<ScoreDoc>comparingDouble(scoreDoc -> scoreDoc.score).reversed().thenComparingInt(scoreDoc -> scoreDoc.doc));
        ScoreDoc[] top = merged.subList(0, Math.min(k, merged.size())).toArray(ScoreDoc[]::new);
        return new TopDocs(new TotalHits(top.length, TotalHits.Relation.EQUAL_TO), top);
    }

    private static KnnVectorsFormat preconditionedFormat() {
        return new ESNextDiskBBQVectorsFormat(
            ESNextDiskBBQVectorsFormat.QuantEncoding.ONE_BIT_4BIT_QUERY,
            64,
            16,
            DenseVectorFieldMapper.ElementType.FLOAT,
            false,
            null,
            1,
            true,
            ESNextDiskBBQVectorsFormat.DEFAULT_PRECONDITIONING_BLOCK_DIMENSION,
            null
        );
    }

    private static KnnVectorsFormat es920Format() {
        return new ES920DiskBBQVectorsFormat(64, 16);
    }

    /** Appends one segment holding {@code vectors} to {@code dir}, written with {@code format}. */
    private static void writeSegment(Directory dir, KnnVectorsFormat format, float[][] vectors) throws IOException {
        IndexWriterConfig config = new IndexWriterConfig().setCodec(TestUtil.alwaysKnnVectorsFormat(format))
            .setMaxBufferedDocs(vectors.length * 2)
            .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH)
            .setMergePolicy(NoMergePolicy.INSTANCE);
        try (IndexWriter writer = new IndexWriter(dir, config)) {
            for (float[] vector : vectors) {
                Document doc = new Document();
                doc.add(new KnnFloatVectorField(FIELD, vector, VectorSimilarityFunction.EUCLIDEAN));
                writer.addDocument(doc);
            }
            writer.commit();
        }
    }

    private static float[][] randomVectors(int count, int dims) {
        float[][] vectors = new float[count][];
        for (int i = 0; i < count; i++) {
            vectors[i] = randomVector(dims);
        }
        return vectors;
    }

    private static float[] randomVector(int dims) {
        float[] vector = new float[dims];
        for (int i = 0; i < dims; i++) {
            vector[i] = randomFloat() * 2 - 1;
        }
        return vector;
    }
}
