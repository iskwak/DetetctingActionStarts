local param_helper = {}

---------------------------------------------------------------------
-- check "parameters"
function param_helper.param_checker(name, param)
    if param == nil then
        error(name .. ' not set');
    end
end


function param_helper.param_list_check(names, param)
    print(names)
    num_params = #names
    for i=1,num_params do
        param_helper.param_checker(names[i], params[names[i]])
    end
end


return param_helper;
