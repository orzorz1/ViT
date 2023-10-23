from segment_anything import sam_model_registry

sam_model = sam_model_registry['vit_b'](checkpoint='../commons/segment-anything/savedModel/sam_vit_b_01ec64.pth')

print(sam_model)