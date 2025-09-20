package proxies

import (
	"fmt"
	"log"
	"net"
	"net/http"
	"strings"
)

func RunReadWriteProxy() error {
	listenSocket := getEnv("LISTEN_SOCKET", defaultRWListenSocket)
	upstreamSocket := getEnv("UPSTREAM_SOCKET", defaultUpstreamSocket)

	if err := ensureParentDir(listenSocket); err != nil {
		return fmt.Errorf("failed to prepare directory for %s: %w", listenSocket, err)
	}
	removeIfExists(listenSocket)

	proxy := newUnixReverseProxy(upstreamSocket, allowReadWrite)

	listener, err := net.Listen("unix", listenSocket)
	if err != nil {
		return fmt.Errorf("failed to listen on %s: %w", listenSocket, err)
	}
	ensureSocketMode(listenSocket, socketPermissions)

	server := &http.Server{Handler: logRequests(proxy)}

	log.Printf("Read-write proxy listening on %s -> %s", listenSocket, upstreamSocket)
	if err := server.Serve(listener); err != nil && err != http.ErrServerClosed {
		return fmt.Errorf("proxy server error: %w", err)
	}
	return nil
}

func allowReadWrite(r *http.Request, body []byte) error {
	if err := allowReadOnly(r, body); err == nil {
		return nil
	}

	path := r.URL.Path
	switch r.Method {
	case http.MethodPost:
		if isCursorPath(path) || strings.Contains(path, "/_api/import") || strings.Contains(path, "/_api/document") || strings.Contains(path, "/_api/collection") || strings.Contains(path, "/_api/index") {
			return nil
		}
	case http.MethodPut, http.MethodPatch, http.MethodDelete:
		if strings.Contains(path, "/_api/document") || strings.Contains(path, "/_api/collection") || strings.Contains(path, "/_api/index") {
			return nil
		}
	}

	return fmt.Errorf("method %s not permitted on %s", r.Method, path)
}
