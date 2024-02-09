wandb login $WANDB_API_KEY

python neurons/miner.py --wallet.name $WALLET_NAME \
                        --wallet.hotkey $HOTKEY_NAME --logging.debug \
                        --axon.port $AXON_PORT --dht.port $DHT_PORT --dht.announce_ip $ANNOUNCE_IP \
                        --netuid 34 --subtensor.network test \