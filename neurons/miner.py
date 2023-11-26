# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Bittensor Miner Template:
# TODO(developer): Rewrite based on protocol and validator defintion.

# Step 1: Import necessary libraries and modules
import os
import time
import argparse
import typing
import traceback
import bittensor as bt

# import this repo
import template
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausFalLM, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

def get_config():
    # Step 2: Set up the configuration parser
    # This function initializes the necessary command-line arguments.
    # Using command-line arguments allows users to customize various miner settings.
    parser = argparse.ArgumentParser()
    # TODO(developer): Adds your custom miner arguments to the parser.
    parser.add_argument('--custom', default='my_custom_value', help='Adds a custom value to the parser.')
    # Adds override arguments for network and netuid.
    parser.add_argument( '--netuid', type = int, default = 1, help = "The chain subnet uid." )
    parser.add_argument( '--axon.port', type = int, default = 8091)
    # Adds subtensor specific arguments i.e. --subtensor.chain_endpoint ... --subtensor.network ...
    bt.subtensor.add_args(parser)
    # Adds logging specific arguments i.e. --logging.debug ..., --logging.trace .. or --logging.logging_dir ...
    bt.logging.add_args(parser)
    # Adds wallet specific arguments i.e. --wallet.name ..., --wallet.hotkey ./. or --wallet.path ...
    bt.wallet.add_args(parser)
    # Adds axon specific arguments i.e. --axon.port ...
    bt.axon.add_args(parser)
    # Activating the parser to read any command-line inputs.
    # To print help message, run python3 template/miner.py --help
    config = bt.config(parser)

    # Step 3: Set up logging directory
    # Logging captures events for diagnosis or understanding miner's behavior.
    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            'miner',
        )
    )
    # Ensure the directory for logging exists, else create one.
    if not os.path.exists(config.full_path): os.makedirs(config.full_path, exist_ok=True)
    return config


# Main takes the config and starts the miner.
def main( config ):

    # Activating Bittensor's logging with the set configurations.
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(f"Running miner for subnet: {config.netuid} on network: {config.subtensor.chain_endpoint} with config:")

    # This logs the active configuration to the specified logging directory for review.
    bt.logging.info(config)

    # Step 4: Initialize Bittensor miner objects
    # These classes are vital to interact and function within the Bittensor network.
    bt.logging.info("Setting up bittensor objects.")

    # Wallet holds cryptographic information, ensuring secure transactions and communication.
    wallet = bt.wallet( config = config )
    bt.logging.info(f"Wallet: {wallet}")

    # subtensor manages the blockchain connection, facilitating interaction with the Bittensor blockchain.
    subtensor = bt.subtensor( config = config )
    bt.logging.info(f"Subtensor: {subtensor}")

    # metagraph provides the network's current state, holding state about other participants in a subnet.
    metagraph = subtensor.metagraph(config.netuid)
    bt.logging.info(f"Metagraph: {metagraph}")

    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        bt.logging.error(f"\nYour validator: {wallet} if not registered to chain connection: {subtensor} \nRun btcli register and try again. ")
        exit()
    else:
        # Each miner gets a unique identity (UID) in the network for differentiation.
        my_subnet_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
        bt.logging.info(f"Running miner on uid: {my_subnet_uid}")

    # Step 4: Set up miner functionalities
    # The following functions control the miner's response to incoming requests.
    # The blacklist function decides if a request should be ignored.
    def blacklist_fn( synapse: template.train.Train ) -> typing.Tuple[bool, str]:
        # TODO(developer): Define how miners should blacklist requests. This Function 
        # Runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        # The synapse is instead contructed via the headers of the request. It is important to blacklist
        # requests before they are deserialized to avoid wasting resources on requests that will be ignored.
        # Below: Check that the hotkey is a registered entity in the metagraph.
        if synapse.dendrite.hotkey not in metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            bt.logging.trace(f'Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}')
            return True, "Unrecognized hotkey"
        # TODO(developer): In practice it would be wise to blacklist requests from entities that 
        # are not validators, or do not have enough stake. This can be checked via metagraph.S
        # and metagraph.validator_permit. You can always attain the uid of the sender via a
        # metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.
        # Otherwise, allow the request to be processed further.
        bt.logging.trace(f'Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}')
        return False, "Hotkey recognized!"

    # The priority function determines the order in which requests are handled.
    # More valuable or higher-priority requests are processed before others.
    def priority_fn( synapse: template.train.Train ) -> float:
        # TODO(developer): Define how miners should prioritize requests.
        # Miners may recieve messages from multiple entities at once. This function
        # determines which request should be processed first. Higher values indicate
        # that the request should be processed first. Lower values indicate that the
        # request should be processed later.
        # Below: simple logic, prioritize requests from entities with more stake.
        caller_uid = metagraph.hotkeys.index( synapse.dendrite.hotkey ) # Get the caller index.
        prirority = float( metagraph.S[ caller_uid ] ) # Return the stake as the priority.
        bt.logging.trace(f'Prioritizing {synapse.dendrite.hotkey} with value: ', prirority)
        return prirority

    # This is the core miner function, which decides the miner's response to a valid, high-priority request.
    def train( synapse: template.train.Train ) -> template.train.Train:
        # TODO(developer): Define how miners should process requests.
        # This function runs after the synapse has been deserialized (i.e. after synapse.data is available).
        # This function runs after the blacklist and priority functions have been called.

        # # Use CUDA if available, otherwise use CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pre-trained model and tokenizer
        # model_name = 'sshleifer/tiny-gpt2'
        model = AutoModelForCausalLM.from_pretrained(synapse.model_name)
        
        for layer, weight in zip(model.parameters(), synapse.model_weights):
            # layer = torch.nn.parameter.Parameter(weight)
            layer = torch.nn.parameter.Parameter(bt.Tensor.deserialize(weight).clone().detach())

        tokenizer = AutoTokenizer.from_pretrained(synapse.model_name)
        
        # Add the EOS token as PAD token to ensure our dataloader doesn't throw an error for sequences of unequal length
        tokenizer.pad_token = tokenizer.eos_token
        # Move the model to the appropriate device
        model.to(device)

        # synapse.gradients = [1,2]

        # Load optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr = synapse.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=synapse.steps)  

        # Load dataset
        dataset = load_dataset(synapse.dataset_name, 'wikitext-2-v1', split='train')#streaming=True? or some lazy loading TODO shard dataset on HF
        dataset = dataset.select(range(synapse.dataset_indices[0], synapse.dataset_indices[1]))

        # Define encoding function
        def encode(examples):
            return tokenizer(examples['text'], truncation=True, max_length=512, padding='max_length', return_tensors='pt')

        # Encode the dataset
        encoded_dataset = dataset.map(encode, batched=True)
        
        # Create a PyTorch DataLoader
        dataloader = DataLoader(encoded_dataset, batch_size=synapse.batch_size)#TODO determined by device capacity

        # Train data for one epoch
        for step, batch in enumerate(dataloader):
            # break
            # Move batch to device
            # input_ids = batch['input_ids'].to(device)
            # attention_mask = batch['attention_mask'].to(device)
            # labels = batch["input_ids"].to(device)

            input_ids = torch.stack(batch['input_ids']).to(device)
            attention_mask = torch.stack(batch['attention_mask']).to(device)
            labels = torch.stack(batch['attention_mask']).to(device)

            # Forward pass
            outputs = model(
                input_ids = input_ids, 
                attention_mask =attention_mask,
                labels = labels
            )     
            
            # Backward pass    
            loss = outputs.loss
            print(step)
            print(loss)
            # synpase.loss = loss
            loss.backward()

            #FIXME does this cause a memory leak?
            if step == 0:
                synapse.gradients = []
                # Store gradients
                for layer in model.parameters():
                    synapse.gradients.append(layer.grad)
            else:
                # Store gradients
                for i, layer in enumerate(model.parameters()):
                    synapse.gradients[i] += layer.grad

            # Adjust gradient
            optimizer.step()
            scheduler.step() 
            optimizer.zero_grad()

            if step == 10:
                break

        for i, layer in enumerate(model.parameters()):
            synapse.gradients[i] = bt.Tensor.serialize(synapse.gradients[i])

        bt.logging.info(f"Final synapse {synapse}")

        outputs = model(
            input_ids = torch.stack(batch["input_ids"]).to(device), 
            attention_mask = torch.stack(batch["attention_mask"]).to(device),
            labels = torch.stack(batch["input_ids"]).to(device)
        )  

        print("loss")
        synapse.loss = float(outputs.loss)
        print(synapse.loss)

        return synapse

    # Step 5: Build and link miner functions to the axon.
    # The axon handles request processing, allowing validators to send this process requests.
    axon = bt.axon( 
        wallet = wallet,
        port=config.axon.port
    )

    bt.logging.info(f"Axon {axon}")

    # Attach determiners which functions are called when servicing a request.
    bt.logging.info(f"Attaching forward function to axon.")
    axon.attach(
        forward_fn = train,
        blacklist_fn = blacklist_fn,
        priority_fn = priority_fn,
    )

    # Serve passes the axon information to the network + netuid we are hosting on.
    # This will auto-update if the axon port of external ip have changed.
    bt.logging.info(f"Serving axon {train} on network: {config.subtensor.chain_endpoint} with netuid: {config.netuid}")
    axon.serve( netuid = config.netuid, subtensor = subtensor )

    # Start  starts the miner's axon, making it active on the network.
    bt.logging.info(f"Starting axon server on port: {config.axon.port}")
    axon.start()

    # Step 6: Keep the miner alive
    # This loop maintains the miner's operations until intentionally stopped.
    bt.logging.info(f"Starting main loop")
    step = 0
    while True:
        try:
            # TODO(developer): Define any additional operations to be performed by the miner.
            # Below: Periodically update our knowledge of the network graph.
            if step % 5 == 0:
                metagraph = subtensor.metagraph(config.netuid)
                log =  (f'Step:{step} | '\
                        f'Block:{metagraph.block.item()} | '\
                        f'Stake:{metagraph.S[my_subnet_uid]} | '\
                        f'Rank:{metagraph.R[my_subnet_uid]} | '\
                        f'Trust:{metagraph.T[my_subnet_uid]} | '\
                        f'Consensus:{metagraph.C[my_subnet_uid] } | '\
                        f'Incentive:{metagraph.I[my_subnet_uid]} | '\
                        f'Emission:{metagraph.E[my_subnet_uid]}')
                bt.logging.info(log)
            step += 1
            time.sleep(1)

        # If someone intentionally stops the miner, it'll safely terminate operations.
        except KeyboardInterrupt:
            axon.stop()
            bt.logging.success('Miner killed by keyboard interrupt.')
            break
        # In case of unforeseen errors, the miner will log the error and continue operations.
        except Exception as e:
            bt.logging.error(traceback.format_exc())
            continue


# This is the main function, which runs the miner.
if __name__ == "__main__":
    main( get_config() )
