# 说明
    1. coco_label.py    COCO数据集处理成 yolo所需格式
    2. copy_category.py 从数据中拷贝特点类别的图片和标签等
    3. drawBox.py       根据标注文件和原图画出bounding box和类名，来检查标注是否正确
    4. evalute.py       分析日志，评估每一类物体的训练检测结果
    5. extract_log.py   提取训练log，去除不可解析的log后使log文件格式化，生成新的log文件供可视化工具绘图
    6. imagenet_to_yolo.py imagenet 数据集 生成 yolo标签格式
    7. k_means_yolo.py  通过k-means ++ 算法获取YOLOv2需要的anchors的尺寸
    8. lableImg_voc_to_yolo.py  VOC数据集 生成 yolo标签格式
    9. make_labels_cn.py        制作coco数据集 中文标签
    10. partial.py       将数据集随机分为训练集和测试集。并提取出来自Paul、COCO和ImageNet的图片列表
    11. statistics.py    统计数据集中每类物体有多少图片和ROI
    12. train_iou_visualization.py    训练日志 iou可视化
    13. train_loss_visualization.py   训练日志 loss可视化
