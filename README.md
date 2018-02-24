# Reproducible Example

Please install [docker](https://docs.docker.com/install/) and use the following commands to re-run the experiments

```sh
docker run -it wush978/deepcensor:latest /bin/bash
# under docker
cd deepcensor
git pull origin master
source bin/activate
cd exp
python train.py --config linear-normal-no-ipinyou.exp.data-201310_1e-4/01.json
```

