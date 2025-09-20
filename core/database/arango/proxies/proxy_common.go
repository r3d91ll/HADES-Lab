package proxies

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"golang.org/x/net/http2"
)

const (
	defaultUpstreamSocket = "/tmp/arangodb.sock"
	defaultROListenSocket = "/run/hades/readonly/arangod.sock"
	defaultRWListenSocket = "/run/hades/readwrite/arangod.sock"
	socketPermissions     = 0o660
)

// UnixReverseProxy forwards HTTP requests to an upstream exposed via Unix socket.
type UnixReverseProxy struct {
	upstreamSocket string
	allowFunc      func(*http.Request, []byte) error
	client         *http.Client
}

func newUnixReverseProxy(upstreamSocket string, allowFunc func(*http.Request, []byte) error) *UnixReverseProxy {
	transport := newUnixTransport(upstreamSocket)
	return &UnixReverseProxy{
		upstreamSocket: upstreamSocket,
		allowFunc:      allowFunc,
		client: &http.Client{
			Transport: transport,
			Timeout:   120 * time.Second,
		},
	}
}

func newUnixTransport(socketPath string) *http.Transport {
	dialer := &net.Dialer{Timeout: 10 * time.Second}
	transport := &http.Transport{
		DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
			return dialer.DialContext(ctx, "unix", socketPath)
		},
		ForceAttemptHTTP2: true,
	}
	if err := http2.ConfigureTransport(transport); err != nil {
		log.Fatalf("failed to configure HTTP/2 transport: %v", err)
	}
	return transport
}

func (p *UnixReverseProxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	bodyBytes, err := readBody(r.Body)
	if err != nil {
		http.Error(w, "failed to read request body", http.StatusBadRequest)
		return
	}

	if err := p.allowFunc(r, bodyBytes); err != nil {
		http.Error(w, err.Error(), http.StatusForbidden)
		return
	}

	upstreamURL := buildUpstreamURL(r)
	upstreamReq, err := http.NewRequestWithContext(r.Context(), r.Method, upstreamURL, bytes.NewReader(bodyBytes))
	if err != nil {
		http.Error(w, "failed to build upstream request", http.StatusInternalServerError)
		return
	}

	copyHeaders(upstreamReq.Header, r.Header)

	resp, err := p.client.Do(upstreamReq)
	if err != nil {
		http.Error(w, fmt.Sprintf("upstream error: %v", err), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	copyHeaders(w.Header(), resp.Header)
	w.WriteHeader(resp.StatusCode)
	if _, err := io.Copy(w, resp.Body); err != nil {
		log.Printf("warning: failed to copy upstream response: %v", err)
	}
}

func readBody(body io.ReadCloser) ([]byte, error) {
	if body == nil {
		return nil, nil
	}
	defer body.Close()
	return io.ReadAll(body)
}

func copyHeaders(dst, src http.Header) {
	for key, values := range src {
		dst.Del(key)
		for _, value := range values {
			dst.Add(key, value)
		}
	}
}

func buildUpstreamURL(r *http.Request) string {
	var builder strings.Builder
	builder.WriteString("http://arangodb")
	builder.WriteString(r.URL.Path)
	if raw := r.URL.RawQuery; raw != "" {
		builder.WriteByte('?')
		builder.WriteString(raw)
	}
	return builder.String()
}

func removeIfExists(path string) {
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		log.Fatalf("failed to remove existing socket %s: %v", path, err)
	}
}

func ensureParentDir(path string) error {
	dir := filepath.Dir(path)
	if dir == "." || dir == "/" {
		return nil
	}
	return os.MkdirAll(dir, 0o750)
}

func ensureSocketMode(path string, mode os.FileMode) {
	if err := os.Chmod(path, mode); err != nil {
		log.Fatalf("failed to chmod %s: %v", path, err)
	}
}

func getEnv(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func logRequests(handler http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		log.Printf("%s %s", r.Method, r.URL.Path)
		handler.ServeHTTP(w, r)
	})
}
