python freeze_graph.py --input_graph=summary/starry_night_graph_def.pb --input_checkpoint=summary/starry_night.ckpt-31000 --output_node_names=image_transform_network/deconv1/output --output_graph=models/starry_night.pb