https://business.yelp.com/data/resources/open-dataset/

https://rpubs.com/limminchim/dsscapstone-005a-v1
https://d396qusza40orc.cloudfront.net/dsscapstone/dataset/yelp_dataset_challenge_academic_dataset.zip

# Differences
Used a different tokenizer. The one used in the paper, Stanford's CoreNLP, requires a running server to be able to call within the Python script.

# Getting Started
Install uv: MacOS/Linux `curl -LsSf https://astral.sh/uv/install.sh | sh` or Windows `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

Create swap memory: https://www.digitalocean.com/community/tutorials/how-to-add-swap-space-on-ubuntu-20-04

`uv run python -m space download en_core_web_sm`