require 'fs'
quacktrain = dofile('modules/quacktrain.lua')

print('reloading embedding_helper');

local embedding_helper = {}

function embedding_helper.create_embedding_lmdb(model, lmdb, outname)
   embedfp = io.open(outname, "w")

   embedfp:write("x,y,clusterid\n")

   num_entries = lmdb:stat().entries
   for i = 1,num_entries do
	  data = lmdb:getData()
	  img,label = loadcaffe.parseCaffeDatum(data)
	  img1 = quacktrain.preproc_img(img[1], img_mean, 1/128, false, false, crop_size)
	  img2 = quacktrain.preproc_img(img[2], img_mean, 1/128, false, false, crop_size)
	  input = {img1:reshape(1,28,28), img2:reshape(1,28,28)}

   end

   embedfp:close()
end


function embedding_helper.create_embedding_imgs(model, imgs, dataset, outname, num_entries, preproc)

   if num_entries == nil then
	  num_entries = imgs:size()[1]
   end
   if preproc == nil then
       preproc = true
   end

   embedfp = io.open(outname, "w")

   --embedfp:write("x,y,clusterid\n")

   for i = 1,num_entries do
	  img = imgs[i]
      if preproc == true then
          img = quacktrain.preproc_img(img, 0, 1, false, true, 28)
      end

	  --embed = model:forward(img)
      embed = model(img)
      --if dataset.files then
      --    embedfp:write(string.format("%f,%f,%d,%s\n",
      --                                embed[1], embed[2], dataset.labels[i]-1,
      --                                dataset.files[i]))
      --else
      --    embedfp:write(string.format("%f,%f,%d\n",
      --                                embed[1], embed[2], dataset.labels[i]-1))
      --end
      --if dataset.recreated_files then
      --    feature_mat:write("," .. dataset.recreated_files[i])
      --end
      for j = 1,embed:size()[1] do
         embedfp:write(string.format("%f,", embed[j]))
      end

	  embedfp:write(string.format("%d", dataset.labels[i]-1))

      if dataset.files then
          embedfp:write("," .. dataset.files[i])
      end
      if dataset.recreated_files then
          embedfp:write("," .. dataset.recreated_files[i])
      end
	  embedfp:write("\n")
   end

   embedfp:close()
end

function embedding_helper.create_original_feature_mat(imgs, dataset, outname, num_entries, preproc)
   feature_mat = io.open(outname, "w")
   print(outname)

   --feature_mat:write("x,y,clusterid\n")

   if num_entries == nil then
	  num_entries = imgs:size()[1]
   end
   if preproc == nil then
       preproc = true
   end

   for i = 1,num_entries do
	  img = imgs[i]
      if preproc == true then
          img = quacktrain.preproc_img(img, 128, 1/128, false, true, 28):float()
          img = img:reshape(img:size()[2]*img:size()[3], 1)

          for j = 1,img:size()[1] do
             feature_mat:write(string.format("%f,", img[j][1]))
          end
      else
          if img:dim() > 1 then
              img = img:reshape(img:size()[2]*img:size()[3], 1)
              for j = 1,img:size()[1] do
                 feature_mat:write(string.format("%f,", img[j][1]))
              end
          else
              for j = 1,img:size()[1] do
                 feature_mat:write(string.format("%f,", img[j]))
              end
          end
      end
 
	  feature_mat:write(string.format("%d", dataset.labels[i]-1))
      if dataset.files then
          feature_mat:write("," .. dataset.files[i])
      end
      if dataset.recreated_files then
          feature_mat:write("," .. dataset.recreated_files[i])
      end
	  feature_mat:write("\n")
   end

   feature_mat:close()
end


return embedding_helper
