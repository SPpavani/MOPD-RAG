# Kubernetes

Kubernetes runs containers in pods placed on nodes. The control plane holds etcd, the API server and the scheduler. The scheduler decides which node runs each new pod, and the kubelet on that node starts it. You describe the desired state in a Deployment and Kubernetes works to keep reality matching it.
