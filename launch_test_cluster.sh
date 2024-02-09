#!/bin/bash

# Define variables here


# Number of containers to run
n_total=5
n_validators=2 #validators are created first, the rest are miners.

# Define a custom network (optional)
network_name="local_test_cpu"
docker network create --subnet=172.18.0.0/16 $network_name

# Base host port number
base_host_axon_port=8000
base_host_dht_port=9000
# Base container port number (if it differs across containers, adjust accordingly)
base_container_axon_port=8000
base_container_dht_port=9000


for ((i=1; i<=n; i++))
do
    # Calculate host and container ports
    host_axon_port=$((base_host_axon_port + i - 1))
    container_axon_port=$((base_container_axon_port + i - 1))

    host_dht_port=$((base_host_dht_port + i - 1))
    container_dht_port=$((base_container_dht_port + i - 1))

    # Run miners
    if [ "$i" -le "$n_validators" ]; then
    docker run -d --name "container_validator_$i" --network $network_name --ip 172.18.0.$i \
               -p $host_axon_port:$container_axon_port -p $host_dht_port:$container_dht_port \
               -e WALLET_NAME=miner_test -e HOTKEY_NAME=miner_hotkey \
               -e AXON_PORT=host_axon_port -e DHT_PORT=host_dht_port \
               validator_image
    else
    docker run -d --name "container_validator_$i" --network $network_name --ip 172.18.0.$i \
               -p $host_axon_port:$container_axon_port -p $host_dht_port:$container_dht_port \
               -e WALLET_NAME=miner_test -e HOTKEY_NAME=miner_hotkey \
               -e AXON_PORT=host_axon_port -e DHT_PORT=host_dht_port \
               miner_image
    fi
    
    
    # Optional: Setup for inter-container communication, configurations, etc.
done

echo "$n containers have been set up and are running."