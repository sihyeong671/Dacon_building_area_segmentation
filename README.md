# Dacon_building_area_segmentation

- mask2former error handling
  - num_classes 수정은 config에서 직접수정해야 적용이 됨(\_base\_ 상속해서 선언 시점에 이미 선언되어 바뀌지 않는 경우 있는 듯함)
  - class weight 수정 (0.5, 1.0)