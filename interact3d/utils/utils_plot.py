import cv2

def plot_bounding_boxes(image, bounding_boxes, color=(0,0,255)):
    """
    image (np.ndarray): Image
    bounding_boxes(np.ndarray or List[List[int]])
    """
    for bb in bounding_boxes:
        xmin, ymin, xmax, ymax = bb
        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), color, 10)
    return image


def plot_point_cloud(points, colors, aX, aY, aZ):
    aX.clear()
    aY.clear()
    aZ.clear()
    
    x,y,z = points[0], points[1], points[2]
    
    aX.scatter(x, y, s=1, alpha=1, c=colors.T)
    aX.scatter(0, 0, s=50, alpha=1, c='red')
    aX.set_xlabel('X')
    aX.set_ylabel('Y')
    aX.set_title(f'Z=0')
    aX.grid(True)
    aX.set_xlim(-0.5, 0.5)
    aX.set_ylim(-0.5, 0.5)
    # aX.set_xlim(-0.11, 0.11)
    # aX.set_ylim(-0.117, 0.114)
    # aX.axis('equal')

    aY.scatter(y, z, s=1, alpha=1, c=colors.T)
    aY.scatter(0, 0, s=50, alpha=1, c='red')
    aY.set_xlabel('Y')
    aY.set_ylabel('Z')
    aY.set_title(f'X=0')
    aY.grid(True)
    aY.set_xlim(-0.5, 0.5)
    aY.set_ylim(-0.5, 0.5)
    # aY.axis('equal')

    aZ.scatter(z, x, s=1, alpha=1, c=colors.T)
    aZ.scatter(0, 0, s=50, alpha=1, c='red')
    aZ.set_xlabel('Z')
    aZ.set_ylabel('X')
    aZ.set_title(f'Y=0')
    aZ.grid(True)
    aZ.set_xlim(-0.5, 0.5)
    aZ.set_ylim(-0.5, 0.5)
    # aZ.set_xlim(-0.11, 0.11)
    # aZ.set_ylim(-0.117, 0.114)
    # aZ.axis('equal')