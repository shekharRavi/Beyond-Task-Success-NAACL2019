def get_spatial_feat(bbox, im_width, im_height):

    x_width = bbox[2]
    y_height = bbox[3]

    x_left = bbox[0]
    x_right = x_left + x_width

    y_upper = im_height - bbox[1]
    y_lower = y_upper - y_height

    x_center = x_left + 0.5*x_width
    y_center = y_lower + 0.5*y_height

    # Rescale features fom -1 to 1

    x_left = (1.*x_left / im_width) * 2 - 1
    x_right = (1.*x_right / im_width) * 2 - 1
    x_center = (1.*x_center / im_width) * 2 - 1

    y_lower = (1.*y_lower / im_height) * 2 - 1
    y_upper = (1.*y_upper / im_height) * 2 - 1
    y_center = (1.*y_center / im_height) * 2 - 1

    x_width = (1.*x_width / im_width) * 2
    y_height = (1.*y_height / im_height) * 2

    # Concatenate features
    feat = [x_left, y_lower, x_right, y_upper, x_center, y_center, x_width, y_height]

    return feat
