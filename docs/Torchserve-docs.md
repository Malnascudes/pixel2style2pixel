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
--extra-files ./scripts/align_all_parallel.py,./utils/common.py,./utils/interpolate.py,./models/psp.py,./models/encoders/psp_encoders.py,./models/encoders/helpers.py,./models/stylegan2/model.py,./models/stylegan2/op/fused_act.py,./models/stylegan2/op/upfirdn2dpkg.py,./configs/paths_config.py,./models/stylegan2/op/upfirdn2d.cpp,./models/stylegan2/op/upfirdn2d_kernel.cu,./models/stylegan2/op/fused_bias_act.cpp,./models/stylegan2/op/fused_bias_act_kernel.cu,./shape_predictor_68_face_landmarks.dat

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

torchserve --start --model-store model_store --models pSp=pSp.mar --ts-config config.properties --ncs
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

# With Docker
```
docker run --rm -it -p 8080:8080 -p 8081:8081 \
            -v $(pwd):/home/model_files \
            --name mar pytorch/torchserve:latest

docker exec -it --user root mar /bin/bash

cd ../model_files
```

# Refs
[How to Serve PyTorch Models with TorchServe Youtube Video](https://www.youtube.com/watch?v=XlO7iQMV3Ik)
[pytroch/serve/examples/image_classifier/resnet_18](https://github.com/pytorch/serve/tree/master/examples/image_classifier/resnet_18)

# upFirdn2d error:
```
TypeError: upfirdn2d(): incompatible function arguments. The following argument types are supported: 1. (arg0: at::Tensor, arg1: at::Tensor, arg2: int, arg3: int, arg4: int, arg5: int, arg6: int, arg7: int, arg8: int, arg9: int) -> at::Tensor
```

This function is used for up-sampling and down-sampling images in the StyleGAN architecture.

[Possible solution](https://github.com/rosinality/stylegan2-pytorch/issues/304) renaming the upfirdn2d.py file to upfirdn2dpkg.py and using PyTorch 1.9.0. However, the user did not provide a clear explanation of why this solution worked. **FUCKING WORKS**

[Another user suggested](https://github.com/sapphire497/style-transformer/issues/12) that the issue might be related to the torch.cpp_extension in the stylegan.op path github.com. This could indicate a problem with the compilation of the custom C++/CUDA code used in StyleGAN.

To troubleshoot this issue, you should confirm that:

* The correct number and type of arguments are being passed to the upfirdn2d() function.
* You are using a compatible version of PyTorch. You might want to try PyTorch 1.9.0, as suggested by the user.
* The custom C++/CUDA code in StyleGAN has been compiled correctly. If you are using a precompiled version of StyleGAN, you might want to try compiling it yourself to ensure that it is compatible with your specific system configuration.

## Change UpFirDn2d.apply inputs
Change inputs from 
```
out = UpFirDn2d.apply(
    input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1]
)
```
to
```
out = UpFirDn2d.apply(
    input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1]
)
```
did not work.

## Check compiled
`Your torch version is 1.6.0 which does not support torch.compile` Message appears, upgrade to compatible version

torch.compile is [only available in pytorch>=2.0](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html). New environment wiht pytorch2.1 created

Same error happens:
```
TypeError: upfirdn2d(): incompatible function arguments. The following argument 1. (arg0: torch.Tensor, arg1: torch.Tensor, arg2: int, arg3: int, arg4: int, arg5: int, arg6: int, arg7: int, arg8: int, arg9: int) -> torch.Tensor
```