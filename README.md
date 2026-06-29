# antigravity-jax-sre
An automated ML SRE built with Google Antigravity for JAX workloads. It uses Multi-Folder Cross-Repository Context to isolate model logic from Pulumi/GCP infrastructure. The agent automates compute and network setup, relying on Interactive Human-in-the-Loop Approval Gates before executing cloud mutations, and enforces Declarative Safety Policies.
