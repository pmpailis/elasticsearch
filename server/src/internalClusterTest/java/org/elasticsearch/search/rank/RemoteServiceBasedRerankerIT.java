/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the Server Side Public License, v 1; you may not use this file except
 * in compliance with, at your election, the Elastic License 2.0 or the Server
 * Side Public License, v 1.
 */

package org.elasticsearch.search.rank;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;

import com.sun.net.httpserver.HttpServer;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.common.util.concurrent.EsExecutors;
import org.elasticsearch.common.util.concurrent.ThreadContext;
import org.elasticsearch.mocksocket.MockHttpServer;
import org.elasticsearch.test.ESIntegTestCase;
import org.elasticsearch.threadpool.ThreadPool;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;

import java.io.IOException;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;

@ESIntegTestCase.ClusterScope(minNumDataNodes = 3)
public class RemoteServiceBasedRerankerIT extends ESIntegTestCase {

    private static final Logger log = LogManager.getLogger(RemoteServiceBasedRerankerIT.class);

    private static HttpServer httpServer;
    private static ExecutorService executorService;
    protected Map<String, HttpHandler> handlers;


    @BeforeClass
    public static void startHttpServer() throws Exception {
        httpServer = MockHttpServer.createHttp(new InetSocketAddress(InetAddress.getLoopbackAddress(), 0), 0);
        ThreadFactory threadFactory = EsExecutors.daemonThreadFactory("[" + RemoteServiceBasedRerankerIT.class.getName() + "]");
        // the EncryptedRepository can require more than one connection open at one time
        executorService = EsExecutors.newScaling(
            RemoteServiceBasedRerankerIT.class.getName(),
            0,
            2,
            60,
            TimeUnit.SECONDS,
            true,
            threadFactory,
            new ThreadContext(Settings.EMPTY)
        );
        httpServer.setExecutor(r -> {
            executorService.execute(() -> {
                try {
                    r.run();
                } catch (Throwable t) {
                    log.error("Error in execution on mock http server IO thread", t);
                    throw t;
                }
            });
        });
        httpServer.start();
    }

    @Before
    public void setUpHttpServer() {
        handlers = new HashMap<>(createHttpHandlers());
        handlers.forEach(httpServer::createContext);
    }

    @AfterClass
    public static void stopHttpServer() {
        httpServer.stop(0);
        ThreadPool.terminate(executorService, 10, TimeUnit.SECONDS);
        httpServer = null;
    }

    @After
    public void tearDownHttpServer() {
        if (handlers != null) {
            for (Map.Entry<String, HttpHandler> handler : handlers.entrySet()) {
                httpServer.removeContext(handler.getKey());
            }
        }
    }

    protected Map<String, HttpHandler> createHttpHandlers() {
        return Collections.singletonMap("/infer", new MockedInferenceHandler());
    }

    static class MockedInferenceHandler implements HttpHandler {

        @Override
        public void handle(HttpExchange exchange) throws IOException {
            try {
                final String request = exchange.getRequestMethod();
            } finally {
                exchange.close();
            }
        }
    }
}
