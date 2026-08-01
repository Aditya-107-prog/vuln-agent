# observability/alloy/Dockerfile
# -----------------------------------------------------------------------------
# Image for the second Railway service ("alloy"). Deploy it by creating a
# new service in the SAME Railway project as vuln-agent (same project =
# same private network) and setting its Root Directory to
# observability/alloy -- otherwise Railway picks up the repo-root
# Dockerfile and you get a second copy of the web app instead.
#
# Pinned rather than :latest so a surprise upstream release can't break a
# redeploy of a working config -- bump deliberately.
FROM grafana/alloy:v1.5.1

# Baked into the image rather than mounted: Railway has no bind mounts,
# so config must ship with the image. Consequence: config changes require
# a redeploy of this service, which is the intended workflow.
COPY config.alloy /etc/alloy/config.alloy

# The upstream image sets an ENTRYPOINT that already invokes alloy, so it
# would prepend itself to the CMD below and produce "alloy alloy run ...".
ENTRYPOINT []

# Shell form so ${PORT} expands -- Railway assigns it at runtime.
#
# --storage.path in /tmp: this is only the remote_write retry WAL, and
# Grafana Cloud is the durable copy. No Railway volume needed.
#
# The HTTP listener is Alloy's own debug UI. It binds 0.0.0.0 (not [::])
# because nothing needs to reach it over the private network -- and you
# should leave this service WITHOUT a public Railway domain so the UI
# stays unreachable from the internet. It exists here mainly so Railway
# sees a listening port.
CMD /bin/alloy run \
      --server.http.listen-addr=0.0.0.0:${PORT:-12345} \
      --storage.path=/tmp/alloy-data \
      /etc/alloy/config.alloy
