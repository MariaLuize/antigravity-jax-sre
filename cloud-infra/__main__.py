import pulumi
import pulumi_gcp as gcp

# Create a VPC Network named jax-gpu-cluster
network = gcp.compute.Network(
    "jax-gpu-network",
    name="jax-gpu-cluster",
    auto_create_subnetworks=False,
    description="VPC for JAX GPU compute cluster"
)

# Create a Subnetwork in us-central1
subnetwork = gcp.compute.Subnetwork(
    "jax-gpu-subnet",
    name="jax-gpu-subnet",
    ip_cidr_range="10.240.0.0/24",
    region="us-central1",
    network=network.id,
    private_ip_google_access=True,
    description="Subnetwork for JAX GPU compute instances"
)

# Create a Firewall Rule to allow SSH access
firewall = gcp.compute.Firewall(
    "jax-gpu-firewall-ssh",
    network=network.id,
    allows=[
        gcp.compute.FirewallAllowArgs(
            protocol="tcp",
            ports=["22"],
        )
    ],
    source_ranges=["0.0.0.0/0"],
    target_tags=["ssh-enabled"],
    description="Allow incoming SSH traffic to instances with ssh-enabled tag"
)

# Create a Compute Engine instance with an NVIDIA T4 GPU in us-central1-a
instance = gcp.compute.Instance(
    "jax-gpu-instance",
    name="jax-gpu-instance",
    machine_type="n1-standard-4",  # N1 machine family is required for Tesla T4 GPUs
    zone="us-central1-a",
    tags=["ssh-enabled"],

    boot_disk=gcp.compute.InstanceBootDiskArgs(
        initialize_params=gcp.compute.InstanceBootDiskInitializeParamsArgs(
            image="debian-cloud/debian-12",
            size=50,  # 50GB disk size to accommodate CUDA and JAX dependencies
        ),
    ),

    network_interfaces=[
        gcp.compute.InstanceNetworkInterfaceArgs(
            network=network.id,
            subnetwork=subnetwork.id,
            # access_configs=[gcp.compute.InstanceAccessConfigArgs()],  # Request public IP
            access_configs=[{}],
        )
    ],

    # Attach NVIDIA Tesla T4 GPU
    guest_accelerators=[
        gcp.compute.InstanceGuestAcceleratorArgs(
            type="nvidia-tesla-t4",
            count=1,
        )
    ],

    # Scheduling configuration is mandatory for instances with attached GPUs
    scheduling=gcp.compute.InstanceSchedulingArgs(
        on_host_maintenance="TERMINATE",
        automatic_restart=True,
    ),

    service_account=gcp.compute.InstanceServiceAccountArgs(
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    ),

    metadata={
        "startup-script": "#!/bin/bash\n# Script can be added here to automate NVIDIA GPU driver installation if desired\n"
    },
)

# Export output values
pulumi.export("network_name", network.name)
# Retrieve the external IP address from the network interface access config
instance_ip = instance.network_interfaces.apply(
    lambda interfaces: interfaces[0].access_configs[0].nat_ip if interfaces[0].access_configs else None
)
pulumi.export("instance_external_ip", instance_ip)
pulumi.export("ssh_command", instance_ip.apply(lambda ip: f"ssh -i ~/.ssh/id_rsa username@{ip}" if ip else "IP not allocated yet"))
