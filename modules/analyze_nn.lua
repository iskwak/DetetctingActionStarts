require 'nn';
require 'cunn';
require 'torch';
require 'cutorch';
require 'inn';
require 'image';
require 'dp';

print("reloading analyze_nn");

local analyze_nn = {};


function analyze_nn.plot_layer(out_dir, layer)

    -- setup output space
    if paths.dirp(out_dir) ~= true then
        paths.mkdir(out_dir)
    end

    local weights,_ = layer:parameters();

    local filters = weights[1]:double()
    local num_filters = filters:size()[1]
    -- make sure the shape makes sense...
    if filters:size():size() == 2 then
        filters = filters:reshape(
            layer.nOutputPlane,
            layer.nInputPlane,
            layer.kW,
            layer.kH)
        --print(filters:size())
    end

    --print(num_filters)
    -- width filters per row?
end


function analyze_nn.save_imagemap(model, params, layer_num, outname)

    local filters1,bias1 = analyze_nn.extract_layer_weights(model, params, layer_num)
    --filters1_dims = torch.(torch.Tensor({1,filters:size()[2]}))
    if model.modules[1].nInputPlane == 3 then
        filters1_dims = 3
    else
        filters1_dims = 1
    end

    local image_map = analyze_nn.build_weight_image(filters1, filters1_dims)
    --image.display(image.toDisplayTensor(image_map))
    image.save(outname, image.toDisplayTensor(image_map))

end



function analyze_nn.build_weight_image(filters, dims)
    -- prob should be parameters
    local filter_img_width = 10
    local filter_width = 100
    local num_filters = filters:size()[1]

    -- start off 256 x 256
    -- want either 3 dimensions or less to display
    local dim_range = torch.range(1,dims):long()

    local image_map = nil --torch.Tensor()
    local curr_row = nil --torch.Tensor()
    local i = 0
    for i = 1,num_filters do
        if i % filter_img_width == 1 then
            if curr_row ~= nil then
                -- curr_row isn't empty means not the first iteration
                -- of the loop. merge this with the image_map
                if image_map == nil then
                    -- image_map is empty
                    image_map = curr_row
                else
                    image_map = torch.cat(image_map, curr_row, 2)
                end
            end

            -- start the first image of the row
            curr_row = image.scale(
			   filters[i]:index(1,dim_range):double(), filter_width, filter_width)
        else
            temp = image.scale(
                filters[i]:index(1,dim_range):double(), filter_width, filter_width)
            -- dim 3 is width
            curr_row = torch.cat(curr_row, temp, 3)
        end
    end
    -- append the remainder
    if i % filter_img_width ~= 1 then 
        if ((filter_width*filter_img_width) - curr_row:size()[3]) > 0 then
            blank = torch.Tensor(
                curr_row:size()[1],
                curr_row:size()[2],
                (filter_width*filter_img_width) - curr_row:size()[3]):fill(0)
            curr_row = torch.cat(curr_row, blank, 3)
        end
        image_map = torch.cat(image_map, curr_row, 2)
    end

    --image.display(image.toDisplayTensor(image_map))
    return image_map
end



function analyze_nn.extract_layer_weights(model, params, layer_num)
    -- loop over the model layers to figure out the correct linear
    -- indexing for the params.
    local curr_idx = 0
    for i = 1,layer_num-1 do
        if model.modules[i].bias ~= nil then
            -- get the dimensions of the current layer
            local bias_size = model.modules[i].bias:size()[1]
            local weight_size = model.modules[i].weight:size()

            local weight_prod = 1
            for j=1,weight_size:size() do
                weight_prod = weight_prod*weight_size[j]
            end

            curr_idx = curr_idx + weight_prod + bias_size
        end
    end

    if model.modules[layer_num].weight == nil then
        print("this layer has no weights")
        return nil,nil
    end

    local weight_size = model.modules[layer_num].weight:size()

    --weight_prod = 1
    --for j=1,weight_size:size() do
    --    weight_prod = weight_prod*weight_size[j]
    --end
    local weight_prod = model.modules[layer_num].nOutputPlane *
            model.modules[layer_num].nInputPlane *
            model.modules[layer_num].kH *
            model.modules[layer_num].kH

    local bias_size = model.modules[layer_num].bias:size()[1]
    --print(weight_size)

    local weight_start = curr_idx + 1
    local weight_end = curr_idx + weight_prod
    local bias_start = curr_idx + weight_prod + 1
    local bias_end = curr_idx + weight_prod + bias_size

    local filters = params:index(1, torch.range(weight_start,weight_end):long()):double()
    local bias = params:index(1, torch.range(bias_start, bias_end):long()):double()

    --print(filters:size())
    --print(torch.type(filters))

    if model.modules[layer_num].__typename ~= 'nn.Linear' then
        -- for now assume only convolutional and linear layers that
        -- need to be reshaped
        --print(filters:size())
        --print(filters:type())
        if string.find(model.modules[layer_num].__typename, 'ccn2') == nil then
        filters = filters:reshape(
            model.modules[layer_num].nOutputPlane,
            model.modules[layer_num].nInputPlane,
            model.modules[layer_num].kH,
            model.modules[layer_num].kW)
        else
        filters = filters:reshape(
            model.modules[layer_num].nInputPlane,
            model.modules[layer_num].kH,
            model.modules[layer_num].kH,
            model.modules[layer_num].nOutputPlane)
            filters = filters:permute(4,1,2,3)
        end
    else
        filters = filters:reshape(
            model.modules[layer_num].parameters:size())
    end

    return filters,bias
end


return analyze_nn;
