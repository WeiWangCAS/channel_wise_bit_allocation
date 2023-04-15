# Channel-Wise Bit Allocation for Deep Visual Feature Quantization 

### Code Implementation

* Parameter Description

  ```python
  	--gpuid: gpu id, type == int
  	--flag: mode selection, type = int
  					flag == 0 is calculate the quantization interval
  					flag == 1 is calculate the quantization error(mse)
  	--model: network models selection, type == str
  					 model == 'alexnet': experimenting with the AlexNet model
             model == 'resnet18': experimenting with the ResNet18 model
             model == 'vgg16': experimenting with the VGGNet16 model
    --quant_mode: quantization method selectiom, type == str
    							quant_mode == 'uniform': use of uniform quantization
    							quant_mode == 'dorefa': use of dorefa quantization
    							quant_mode == 'log': use of logarithmic quantization
    --net_mode: quantization interval table is saved in 												 				'./quant_intval_{net_mode}', type == str
    						model == 'alexnet'	==> net_mode == 'alexnet'
    						model == 'resnet18'	==> net_mode == 'resnet'
    						model == 'vgg16'		==> net_mode == 'vgg'
    --reserve_interval: reserved interval is used for online iterative											 		inference, type == int
    --alpha: weight of mean priority, type == int
    --beta: weight of std priority, type == int
    --csv_path: path for quantization interval table
    --csv_name: name of quantization interval table
  ```

  For example, use gpuid 1 to calculate the uniform quantization interval table of alexnet is:

  ```python
  	python main_model.py --gpuid 1 --flag 0 --model 'alexnet' --quant_mode 'uniform' --csv_path './channel_allocation/' --csv_name 'alexnet'
  ```

  For example, use gpuid 0 to calculate the channel-wise bit allocation quantization error of resnet18 and reserved interval is 20 for online iterative inference and weight of mean priority is 0.9, weight of std priority is 0.1:

  ```python
  	python main_model.py --gpuid 0 --flag 1 --model 'resnet18' --quant_mode 'unniform' --net_mode 'resnet' --reserve_interval 20 --alpha 0.9 --beta 0.1
  ```

* Related Function Descriptions

  algorithm_wrap.py

  ```python
  	def channel_wise_bit_allocation_algorithm(
  			flag, bits, gpuid, 
  			x, net_structure, quant_mode, net_mode, 
  			reserve_interval, alpha, beta,
  			csv_path, csv_name,
  			dorefa_clip_value, 
  			fun_network_forward_inference
  	):
  			'''
        x: extracted intermediate feature
  			net_structure: back-end network structure
  			dorefa_clip_value: dorefa quantization max clip value
  			fun_network_forward_inference: performing inference for back-end 																			 networks
  			'''
        return output # flag == 0 ==> output == 0
      								# flag == 1 ==> output == mse
  ```

* Notes

  - if calculating uniform quantization interval table, please use uniform quantization interval tablefor quantization error(mse), do the same for logarithmic quantization.
  - if calculating the quantization error for alexnet, please e sure that the corrent quantiztaion interval table is in './quant_interval_alexnet', and the same for other network model.
  - The calculated quantization interval table is saved in './channel_allcoation'. Need to move to './quant_interval_{}' when used. 

### Experiment Result

Experiments were conducted at multiple layers of the AlexNet, VggNet, and ResNet network models.

The detailed experimental results are shown in the './experiment_result' folder.

### Environment

```
    Ubuntu20.04
    Python3.9
    Torch1.9.1
    Torchvision0.10.1
    function.so				--> function.cpython-39-x86_64-linux-gnu.so
    algorithm.so			--> algorithm.cpython-39-x86_64-linux-gnu.so
    algorithm_wrap.so	--> algorithm_wrap.cpython-39-x86_64-linux-gnu.so
```