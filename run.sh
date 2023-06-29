# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
set -e
set -x

virtualenv -p python3 .
source ./bin/activate

pip install -r policy_eval/requirements.txt
python -m policy_eval.create_dataset --logtostderr --env_name=HalfCheetah-v2 --models_dir=policy_eval/data --num_episodes=2
python -m policy_eval.train_eval --logtostderr --env_name=HalfCheetah-v2 --num_mc_episodes=1 --num_updates=2
