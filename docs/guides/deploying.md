# 部署指南

- **Docker**：`docker compose up --build`（暴露 8000）。
- **K8s**：`kubectl apply -f k8s/`（Deployment + Service）。
- **Helm**：`helm install lesforge helm/lesforge`。
- **CI**：`.github/workflows/ci.yml` 跑 compileall + unittest + doctor + docker build。

服务默认绑定 `127.0.0.1`，生产环境如需对外暴露，改 `serving.host` 或用 Service 转发。
