import cv2
import numpy as np
from random import randint
from functools import reduce

def nothing(arg):
    return

def show_webcam(mirror=False):
    cv2.namedWindow('Webcam')
    cv2.namedWindow('Detection')
    cv2.createTrackbar('Gx', 'Webcam', 100, 500, nothing)
    cv2.createTrackbar('Gy', 'Webcam', 200, 500, nothing)
    cv2.createTrackbar('Raw|Edges', 'Webcam', 0, 1, nothing)
    cv2.createTrackbar('minRadius', 'Detection', 10, 500, nothing)
    cv2.createTrackbar('maxRadius', 'Detection', 200, 500, nothing)
    #cv2.createTrackbar('smartDetection', 'Detection', 0, 10, nothing)

    cam = cv2.VideoCapture(0)
    #tryCount = 20
    hist_bucket = []
    false_counter = 0
    while True:
        ret_val, img = cam.read()
        edges = None
        circles = None

        gx = cv2.getTrackbarPos('Gx', 'Webcam')
        gy = cv2.getTrackbarPos('Gy', 'Webcam')
        minRadius = cv2.getTrackbarPos('minRadius', 'Detection')
        maxRadius = cv2.getTrackbarPos('maxRadius', 'Detection')
        raw_img = cv2.getTrackbarPos('Raw|Edges', 'Webcam')
        #smart_detection = cv2.getTrackbarPos('smartDetection', 'Detection')
        if(maxRadius == 0):
            maxRadius = 1
        

        if mirror: 
            img = cv2.flip(img, 1)
        '''
        if smart_detection == 0:
            edges = display_edges(img, gx=gx, gy=gy)
            circles = detect_circles(img, edges, minRadius=minRadius, maxRadius=maxRadius)
        else:
            circles = mc_sim_circles(goal=smart_detection, image=img, gx=gx, gy=gy, minRadius=minRadius, maxRadius=maxRadius, maxTries=tryCount) or img
        '''
        edges = display_edges(img, gx=gx, gy=gy)
        #circles = detect_circles(img, edges, minRadius=minRadius, maxRadius=maxRadius)
        circles, raw_circs = detect_circle_contours(img, edges)
        circles = detect_circles(img, raw_circs, minRadius=minRadius, maxRadius=maxRadius, maxCircles=1, hist_bucket=hist_bucket, false_counter=false_counter)
        cv2.setTrackbarPos('maxRadius', 'Detection', maxRadius)
        cv2.setTrackbarPos('minRadius', 'Detection', minRadius)
        cv2.setTrackbarPos('Gx', 'Webcam', gx)
        cv2.setTrackbarPos('Gy', 'Webcam', gy)
        '''
        if tryCount == 0:
            tryCount = 200
            cv2.setTrackbarPos('smartDetection', 'Detection', 0)
        '''
        cv2.imshow('Webcam', img if raw_img==0 else edges)
        cv2.imshow('Detection', circles)
        if false_counter > 100:
            print("Too many false positives ignored, resetting buckets...")
            hist_bucket = []
            false_counter = 0
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        
def mc_sim_circles(goal=1, image=None, gx=100, gy=200, minRadius=10, maxRadius=500, maxTries=100):
    new_img = image.copy

    for attempt in range(maxTries):
        gx = randint(1, 500)
        gy = randint(1, 500)
        minRadius = randint(1, 500)
        maxRadius = randint(1, 500)

        edges = cv2.Canny(image, gx, gy)
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1,20, param1=50, param2=100, minRadius=minRadius, maxRadius=maxRadius)
        if circles is not None and len(circles) >= goal:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(new_img, (x,y), r, (255, 255, 255), 4)
                cv2.rectangle(new_img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            return new_img
        else:
            continue
        maxTries = maxTries - 1

    return None

def display_edges(img, gx=100, gy=200):
    edges = cv2.Canny(img, gx, gy)
    return edges

def detect_circles(img, edges, minRadius=10, maxRadius=500, maxCircles=1, hist_bucket=[], false_counter=0):
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1,20, param1=50, param2=100, minRadius=minRadius, maxRadius=maxRadius)
    new_img = img.copy()
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        print("Found %s circles" % len(circles))
        if len(circles) > maxCircles:
            def drop_circs(circles, maxCircles):
                new_circs = []
                try:
                    while len(new_circs) < maxCircles:
                        max_r = 0
                        for i in range(len(circles)):
                            x,y,r = circles[i]
                            if r>max_r:
                                max_r = r
                                new_circs.append((x,y,r))
                                circles = np.delete(circles, i)
                except:
                    nothing(None)
                return new_circs
            circles = drop_circs(circles, maxCircles)
        for (x, y, r) in circles:
            if r < 1:
                continue
            #print("Circle pixel val %s" % new_img[y][x])
            if not filter_circle_pixel(new_img, r, y, x, hist_bucket=hist_bucket, false_count=false_counter):
                cv2.circle(new_img, (x,y), r, (255, 255, 255), 4)
                cv2.rectangle(new_img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            else:
                cv2.circle(new_img, (x,y), r, (0, 255, 255), 4)
                cv2.rectangle(new_img, (x - 5, y - 5), (x + 5, y + 5), (128, 255, 128), -1)
            hist_bucket.append((0, new_img[y][x]))
    return new_img

def filter_circle_pixel(img, r, y, x, hist_bucket=[], steps=100, false_count=0):
    #drop values that aren't within a tolerance of the previous close color and radius
    def find_in_bucket(pixel, radius, hist_bucket, steps, color_tolerance, radius_tolerance):
        def minus_perc(x, perc):
            return x - (float(x)*perc)
        def plus_perc(x, perc):
            return x + (float(x)*perc)
        def in_range(x, range):
            print("%s >= %s >= %s" % (range[0], x, range[1]))
            return x <= range[1] and x >= range[0]
        
        if len(hist_bucket) < steps:
            return None

        r_range = (minus_perc(pixel[0], color_tolerance),plus_perc(pixel[0], color_tolerance))
        g_range = (minus_perc(pixel[1], color_tolerance),plus_perc(pixel[1], color_tolerance))
        b_range = (minus_perc(pixel[2], color_tolerance),plus_perc(pixel[2], color_tolerance))
        radius_range = (minus_perc(radius, radius_tolerance), plus_perc(radius, radius_tolerance))
        ret_cap = None
        for i, (hits, capture) in reversed(list(enumerate(hist_bucket))):
            if in_range(capture[0], r_range) and in_range(capture[1], g_range) and in_range(capture[2], b_range):
                print("Hit! %s" % capture)
                hits+=1
                ret_cap = capture
                break
            else:
                print("Miss! %s" % capture)
                hits-=1
            if i >= steps:
                break
        #prune bad data
        try:
            for i, (hits, capture) in list(enumerate(hist_bucket)):
                if hits <=0:
                    print("Dropping capture with %s hits: %s" % (hits, capture))
                    del hist_bucket[i]
        except:
            nothing(None)
        return ret_cap
    pixel = None
    try:
        pixel = img[y][x]
    except:
        return True
    capture = find_in_bucket(pixel,r, hist_bucket, steps, 0.25, 0.25)
    if len(hist_bucket) <= steps:
        return False
    if capture:
        return False
    return True

def detect_circle_contours(img, edges):
    new_img = img.copy()
    dup_edges = edges.copy()
    retval, image = cv2.threshold(dup_edges, 50, 255, cv2.THRESH_BINARY)
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    image = cv2.dilate(dup_edges, el, iterations=6)
    contours, hierarchy = cv2.findContours(
        image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    centers = []
    radii = []

    #print("calculating contour centers")
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            #print("Skipping contour %s" % contour)
            continue
        br = cv2.boundingRect(contour)
        if br[2]:
            #print("Adding radii %s" % br[2])
            if radii is None:
                radii = []
            radii = radii.append(br[2])
        else:
            continue

        m = cv2.moments(contour)
        center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
        centers.append(center)
    if radii:
        radius = int(np.average(radii)) + 5
        print("Total circles found: %s" % len(centers))
        for center in centers:
            cv2.circle(new_img, center, 3, (255, 0, 0), -1)
            cv2.circle(new_img, center, radius, (0, 255, 0), 1)
    return new_img, dup_edges

if __name__ == '__main__':
    show_webcam(mirror=True)