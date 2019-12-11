require 'nn'
require 'optim'
require 'lmdb'
require 'image'
--require 'analyze_nn'
dofile('modules/analyze_nn.lua')

print('reloading quacktrain');

local quacktrain = {};


-- base closure for optim
-- x represents the weight vector
function quacktrain.feval(x, parameters, model, gradParameters, inputs,
						  targets, criterion, batch_size)
   -- get new parameters
   if x ~= parameters then
	  parameters:copy(x)
   end
   if batch_size == nil then
	  batch_size = #inputs
   end

   -- reset gradients
   gradParameters:zero()

   -- f is the average of all criterions
   local outputs = model:forward(inputs)
   local err = criterion:forward(outputs, targets)
   local df_do = criterion:backward(outputs, targets)
   model:backward(input, df_do)


   -- normalize gradients and f(X)
   gradParameters:div(batch_size)
   f = err/batch_size
   --f = err

   -- return f and df/dX
   return f,gradParameters
end


-- Use a typical generic gradient update function
function quacktrain.gradUpdate(mlp, x, y, criterion, learningRate)
    local pred = mlp:forward(x)
    local err = criterion:forward(pred, y)
    local gradCriterion = criterion:backward(pred, y)
    mlp:zeroGradParameters()
    mlp:backward(x, gradCriterion)
    mlp:updateParameters(learningRate)

    return err
end


function quacktrain.plot_msg(msg)
   --print("[" .. timer:time().real .. "] " .. msg)
   print("[" .. os.date() .. "] " .. msg)
end


-- standard preprocessing for nueral network training
function quacktrain.preproc_img(img, mean, scale, is_train, to_crop, crop_size)
   local h,w,temp
   --local crop_size = 224
   if to_crop == true then
	  if is_train == true then
		 h = torch.floor(torch.uniform(1,img:size()[2]-crop_size))
		 w = torch.floor(torch.uniform(1,img:size()[3]-crop_size))
		 --h = math.floor((img:size()[2] - crop_size)/2)
		 --w = math.floor((img:size()[3] - crop_size)/2)
	  else
		 h = (img:size()[2] - crop_size)/2
		 w = (img:size()[3] - crop_size)/2
	  end
	  temp = img[{ {}, {h+1,h+crop_size}, {w+1,w+crop_size}}]
   else
	  temp = img
   end
   --print(temp:size())
   --local temp = img 

   if torch.round(torch.uniform()) == 1 and is_train == true then
       --print(image.hflip)
       --print(temp:size())
	  --temp = image.hflip(temp:float())
   end

   temp = ((temp - mean)*scale):cuda()

   return temp
end


function quacktrain.compute_dataset_mean(mean_cache, dbcursor, num_entries)
   --print(mean_cache)
   local mean = 0
   if mean_cache == nil then
	  mean_cache = ''
   end
   if paths.filep(mean_cache) == false then
	  quacktrain.plot_msg("Calculating Mean")

	  -- get image extents
	  local data = dbcursor:getData()
	  local img,label = loadcaffe.parseCaffeDatum(data)
	  local num_pixels = 1
	  for i = 1,img:size():size() do
		 num_pixels = num_pixels * img:size()[i]
	  end
	  img = nil
	  data = nil

	  for i = 1,num_entries do
		 if i % 1000 == 0 then
			collectgarbage()
			quacktrain.plot_msg(i .. ' of ' .. num_entries)
		 end
		 --batch = trainSet:index(batch, torch.Tensor(1):fill(i):long())
		 local data = dbcursor:getData()
		 local img,label = loadcaffe.parseCaffeDatum(data)
		 mean = mean + img:sum()
		 local label = label + 1
	  end

	  mean = mean/(num_entries*num_pixels)
	  if mean_cache ~= '' then
		 torch.save(mean_cache, mean)
	  end
   else
	  quacktrain.plot_msg("Loading mean..")
	  mean = torch.load(mean_cache)
	  quacktrain.plot_msg("done!")
   end
   return mean
end


function quacktrain.load_readdb(name, db_path)
   local db = lmdb.env({
		 Path = db_path,
		 Name = name})

   db:open()
   local reader = db:txn(true)
   local cursor = reader:cursor()

   return db,cursor
end


function quacktrain.save_weights(model, parameters, layer_num, weight_dims, out_dir, iters)

   local weights,bias = analyze_nn.extract_layer_weights(model, parameters, layer_num)

   local outname = string.format(
	  '%s/%08d_min%f-max%f.png',
	  out_dir,
	  iters,
	  weights:min(),
	  weights:max())

   if model.modules[layer_num].nInputPlane == 3 then
	  weight_dims = 3
   else
	  weight_dims = 1
   end

   local image_map = analyze_nn.build_weight_image(weights, weight_dims)
   image.save(outname, image.toDisplayTensor(image_map))

   quacktrain.plot_msg("\tSaving weight map: " .. outname)
end


function quacktrain.check_validation_error(valcursor, criterion, test_size, mean, img_scale, do_crop, crop_size)
   quacktrain.plot_msg("\tChecking accuracy")
   local f = 0
   local correct = 0
   for k=1,test_size do
	  local data = valcursor:getData()
	  local img,label = loadcaffe.parseCaffeDatum(data)
	  label = label + 1

	  img = quacktrain.preproc_img(img, mean, img_scale, false, do_crop, crop_size)

	  local pred = model:forward(img)

	  local err = criterion:forward(pred, label[1])
	  f = f + err

	  local val,idx = pred:exp():max(1)
	  if idx[1] == label[1] then
		 correct = correct + 1
	  end

	  if valcursor:next() == false then
		 valcursor:first()
	  end
   end
   quacktrain.plot_msg("\tValid Accuracy: " .. correct/test_size)
   quacktrain.plot_msg("\tValid Train Error: " .. f/test_size)

   return f/test_size, correct/test_size
end


function quacktrain.setup_outputs(base_out_dir)
   if paths.dirp(base_out_dir) ~= true then
	  paths.mkdir(base_out_dir)
   end
   filter_dir = paths.concat(base_out_dir, 'filters')
   if paths.dirp(filter_dir) ~= true then
	  paths.mkdir(filter_dir)
   end
   grad_dir = paths.concat(base_out_dir, 'grads')
   if paths.dirp(grad_dir) ~= true then
	  paths.mkdir(grad_dir)
   end
   state_dir = paths.concat(base_out_dir, 'states')
   if paths.dirp(state_dir) ~= true then
	  paths.mkdir(state_dir)
   end

   return filter_dir,grad_dir,state_dir
end



function quacktrain.set_multipliers(model, params, weight_mult, bias_mult)
   -- loop over the model layers to figure out the correct linear
   -- indexing for the params.
   local multiplier = torch.Tensor(params:size()):fill(0)
   local curr_idx = 1
   for i = 1,#model.modules do
	  if model.modules[i].bias ~= nil then
		 -- get the dimensions of the current layer
		 local bias_size = model.modules[i].bias:size()[1]
		 local weight_size = model.modules[i].weight:size()

		 local weight_prod = 1
		 for j=1,weight_size:size() do
			weight_prod = weight_prod*weight_size[j]
		 end

		 max_idx = curr_idx + weight_prod - 1
		 while curr_idx <= max_idx do
			multiplier[curr_idx] = weight_mult
			curr_idx = curr_idx + 1
		 end
		 max_idx = curr_idx + bias_size - 1
		 while curr_idx <= max_idx do
			multiplier[curr_idx] = bias_mult
			curr_idx = curr_idx + 1
		 end

	  end
   end

   return multiplier
end


return quacktrain;
