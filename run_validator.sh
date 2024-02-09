wandb login $WANDB_API_KEY

python neurons/validator.py --wallet.name $WALLET_NAME \
                        --wallet.hotkey $HOTKEY_NAME --logging.debug \
                        --neuron.initial_peers \ 
                        --axon.port $AXON_PORT --dht.port $DHT_PORT --dht.announce_ip $ANNOUNCE_IP \
                        --netuid 34 --subtensor.network test \

                    