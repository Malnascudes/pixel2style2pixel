# Install torchserve

```
pip install nvgpu
# Minimum required pytorch 1.5
pip install torchserve torch-model-archiver torch-workflow-archiver

# Install captum https://github.com/pytorch/serve/issues/966
pip install captum
```

Java OpenJDK11 minimum required
```
# Check version
java -version

# Install if needed
sudo apt-get install openjdk-11-jdk

# Change Java version if needed https://www.baeldung.com/linux/java-choose-default-version
sudo update-alternatives --config java
``` 


# Generate Model Archiver

All dependencies (extra files) must be added manualy to the Model Archiver using the `--extra-files` argument. They will be placed at the top folder, making it necessary to change the routes of the files in the imports.

```
torch-model-archiver --model-name pSp --version 1.0 \
--model-file ./models/psp.py \
--serialized-file ./pretrained_models/psp_ffhq_encode.pt \
--handler mean_image_handler \
--requirements requirements.txt \
--extra-files ./scripts/align_all_parallel.py,./utils/common.py,./utils/interpolate.py,./models/psp.py,./models/encoders/psp_encoders.py,./models/encoders/helpers.py,./models/stylegan2/model.py,./models/stylegan2/op/fused_act.py,./models/stylegan2/op/upfirdn2d.py,./configs/paths_config.py,./models/stylegan2/op/upfirdn2d.cpp,./models/stylegan2/op/upfirdn2d_kernel.cu,./models/stylegan2/op/fused_bias_act.cpp,./models/stylegan2/op/fused_bias_act_kernel.cu,./shape_predictor_68_face_landmarks.dat

mkdir model_store
mv pSp.mar model_store/
```

# Run TorchServe

## Stop anydesk
```
java.io.IOException: Failed to bind to address 0.0.0.0/0.0.0.0:7070
```

Same error in: https://discuss.pytorch.org/t/torchserve-stopped-failed-to-bind-address-already-in-use/115232

```
sudo lsof -i:7070
systemctl status anydesk.service
systemctl stop anydesk.service
```

## Start TorchServe
```
torchserve --start --model-store model_store --models pSp=pSp.mar
```


# Test model running
```
curl http://localhost:8081/models
```

# Run Inference
```
curl http://127.0.0.1:8080/predictions/pSp -T <path-to-image>
```

# Stop TorchServe
```
torchserve --stop
```

# Refs
[How to Serve PyTorch Models with TorchServe Youtube Video](https://www.youtube.com/watch?v=XlO7iQMV3Ik)
[pytroch/serve/examples/image_classifier/resnet_18](https://github.com/pytorch/serve/tree/master/examples/image_classifier/resnet_18)
