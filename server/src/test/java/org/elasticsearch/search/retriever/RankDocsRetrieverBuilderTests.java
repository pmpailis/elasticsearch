/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the Server Side Public License, v 1; you may not use this file except
 * in compliance with, at your election, the Elastic License 2.0 or the Server
 * Side Public License, v 1.
 */

package org.elasticsearch.search.retriever;

import org.elasticsearch.index.query.QueryBuilder;
import org.elasticsearch.index.query.RandomQueryBuilder;
import org.elasticsearch.search.builder.SearchSourceBuilder;
import org.elasticsearch.search.rank.RankDoc;
import org.elasticsearch.search.rank.TestRankDoc;
import org.elasticsearch.search.retriever.rankdoc.RankDocsSortBuilder;
import org.elasticsearch.test.ESTestCase;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import static org.elasticsearch.search.vectors.KnnSearchBuilderTests.randomVector;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.instanceOf;

public class RankDocsRetrieverBuilderTests extends ESTestCase {

    private Supplier<RankDoc[]> rankDocsSupplier() {
        final int rankDocsCount = randomIntBetween(0, 10);
        final int shardIndex = 0;
        TestRankDoc[] rankDocs = new TestRankDoc[rankDocsCount];
        int docId = 0;
        for (int i = 0; i < rankDocsCount; i++) {
            TestRankDoc testRankDoc = new TestRankDoc(docId, randomFloat(), shardIndex);
            docId += randomInt(100);
            rankDocs[i] = testRankDoc;
        }
        return () -> rankDocs;
    }

    private List<RetrieverBuilder> innerRetrievers() {
        List<RetrieverBuilder> retrievers = new ArrayList<>();
        int numRetrievers = randomIntBetween(1, 10);
        for (int i = 0; i < numRetrievers; i++) {
            if (randomBoolean()) {
                StandardRetrieverBuilder standardRetrieverBuilder = new StandardRetrieverBuilder();
                standardRetrieverBuilder.queryBuilder = RandomQueryBuilder.createQuery(random());
                if (randomBoolean()) {
                    standardRetrieverBuilder.preFilterQueryBuilders = preFilters();
                }
                retrievers.add(standardRetrieverBuilder);
            } else {
                KnnRetrieverBuilder knnRetrieverBuilder = new KnnRetrieverBuilder(
                    randomAlphaOfLength(10),
                    randomVector(randomInt(10)),
                    null,
                    randomInt(10),
                    randomIntBetween(10, 100),
                    randomFloat()
                );
                if (randomBoolean()) {
                    knnRetrieverBuilder.preFilterQueryBuilders = preFilters();
                }
                retrievers.add(knnRetrieverBuilder);
            }
        }
        return retrievers;
    }

    private List<QueryBuilder> preFilters() {
        List<QueryBuilder> preFilters = new ArrayList<>();
        int numPreFilters = randomInt(10);
        for (int i = 0; i < numPreFilters; i++) {
            preFilters.add(RandomQueryBuilder.createQuery(random()));
        }
        return preFilters;
    }

    private RankDocsRetrieverBuilder createRandomRankDocsRetrieverBuilder() {
        return new RankDocsRetrieverBuilder(
            randomInt(100),
            innerRetrievers(),
            rankDocsSupplier(),
            preFilters());
    }

    public void testBasic() {
        RankDocsRetrieverBuilder retriever = createRandomRankDocsRetrieverBuilder();
        assertEquals(RankDocsRetrieverBuilder.NAME, retriever.getName());
    }

    public void testExtractToSearchSourceBuilder(){
        RankDocsRetrieverBuilder retriever = createRandomRankDocsRetrieverBuilder();
        SearchSourceBuilder source = new SearchSourceBuilder();
        retriever.extractToSearchSourceBuilder(source, randomBoolean());
        assertThat(source.sorts().size(), equalTo(1));
        assertThat(source.sorts().get(0), instanceOf(RankDocsSortBuilder.class));


    }
}
