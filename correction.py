

# function to generate fixations at each word
def generate_fixations_left_skip_regression(aois_with_tokens):
    fixations = []
    word_count = 0
    skip_count = 0
    regress_count = 0
    
    aoi_list = aois_with_tokens.values.tolist()
    
    index = 0
    
    while index < len(aoi_list):
        x, y, width, height, token = aoi_list[index][2], aoi_list[index][3], aoi_list[index][4], aoi_list[index][5], aoi_list[index][7]
        
        word_count += 1
        
        fixation_x = x + width / 3 + random.randint(-10, 10)
        fixation_y = y + height / 2 + random.randint(-10, 10)

        last_skipped = False

        # skipping
        if len(token) < 5 and random.random() < 0.3:
            skip_count += 1 # fixations.append([fixation_x, fixation_y])
            last_skipped = True
        else:
            fixations.append([fixation_x, fixation_y, len(token) * 50])
            last_skipped = False
        
        # regressions    
        if  random.random() > 0.96:
            index -= random.randint(1, 10)

            if index < 0:
                index = 0

            regress_count += 1
        
        index += 1
            
    
    skip_probability = skip_count / word_count
    
    return fixations


# write a function generate offset error as described in the paper
def error_offset(x_offset, y_offset, fixations):
    '''creates error to move fixations (shift in dissertation)'''
    
    pass


# noise
import random

def error_noise(y_noise_probability, y_noise, duration_noise, fixations):
    '''creates a random error moving a percentage of fixations '''
    
    results = []
    
    for fix in fixations:

        x, y, duration = fix[0], fix[1], fix[2]

        # should be 0.1 for %10
        duration_error = int(duration * duration_noise)

        duration += random.randint(-duration_error, duration_error)

        if duration < 0:
            duration *= -1
        
        if random.random() < y_noise_probability:
            results.append([x, y + random.randint(-y_noise, y_noise), duration])
        else:
            results.append([x, y, fix[2]])
    
    return results

# shift

def error_slope(y_slope, fixations):
    ''' creates a downward slope of fixation errors '''
    result = []

    slope = 0

    for fix in fixations:

        x, y = fix[0], fix[1]

        
        result.append([x,y+slope, fix[2]])

        slope += y_slope

    return result


def error_shift(y_shift, line_limit, fixations):

    ''' creates a downward shift of fixation errors '''
    result = []

    shift = 0

    line = 0

    for fix in fixations:
        x, y = fix[0], fix[1]

        if line >= line_limit:
            line = 0
            result.append([x,y+shift, fix[2]])
        else:
            result.append([x, y, fix[2]])

        shift += y_shift
    
    return result

from collections import defaultdict

def map_lines(lines, fixations):

    result = defaultdict(list)

    for fix in fixations:
        result[min(lines, key=lambda x:abs(x-fix[1]))].append(fix)
    
    return result


def error_within_line_regress(x_shift_probability, shift_duration, lines):

    # map fixation to line
    # for each line, have a probability that there will be a regression

    result = []


    for line in lines:

        words = []

        for fix in line[0]:
            x,y, duration = fix[0], fix[1], fix[2]

            words.append(x)
            duration_error = int(duration * shift_duration)

            duration += random.randint(-duration_error, duration_error)

            if duration < 0:
                duration *= -1
            
            if random.random() < x_shift_probability:
                result.append([words[random.randint(0, len(words))], y, duration])
            else:
                result.append(x,y,fix[2])

    return result

def error_between_line_regress(shift_probability, shift_duration, lines):
    
    result = []

    for line in lines:
        for fix in line[0]:

            x, y, duration = fix[0], fix[1], fix[2]

            duration_error = int(duration * shift_duration)

            duration += random.randint(-duration_error, duration_error)

            if duration < 0:
                duration *= -1
            
            if random.random() < shift_probability:
                result.append(x, lines[random.randint(0, len(lines))][0][1], duration)
            else:
                result.append(x,y,fix[2])

    return result


    
        





# droop

from PIL import ImageFont, ImageDraw, Image
from matplotlib import pyplot as plt
import numpy as np


def draw_fixation(Image_file, fixations):
    """Private method that draws the fixation, also allow user to draw eye movement order

    Parameters
    ----------
    draw : PIL.ImageDraw.Draw
        a Draw object imposed on the image

    draw_number : bool
        whether user wants to draw the eye movement number
    """

    im = Image.open(Image_file)
    draw = ImageDraw.Draw(im, 'RGBA')

    if len(fixations[0]) == 3:
        x0, y0, duration = fixations[0]
    else:
        x0, y0 = fixations[0]

    for fixation in fixations:
        
        if len(fixations[0]) == 3:
            duration = fixation[2]
            if 5 * (duration / 100) < 5:
                r = 3
            else:
                r = 5 * (duration / 100)
        else:
            r = 8
        x = fixation[0]
        y = fixation[1]

        bound = (x - r, y - r, x + r, y + r)
        outline_color = (50, 255, 0, 0)
        fill_color = (50, 255, 0, 220)
        draw.ellipse(bound, fill=fill_color, outline=outline_color)

        bound = (x0, y0, x, y)
        line_color = (255, 155, 0, 155)
        penwidth = 2
        draw.line(bound, fill=line_color, width=5)

        x0, y0 = x, y

    plt.figure(figsize=(17, 15))
    plt.imshow(np.asarray(im), interpolation='nearest')


def draw_correction(Image_file, fixations, match_list):
    """Private method that draws the fixation, also allow user to draw eye movement order

    Parameters
    ----------
    draw : PIL.ImageDraw.Draw
        a Draw object imposed on the image

    draw_number : bool
        whether user wants to draw the eye movement number
    """

    im = Image.open(Image_file)
    draw = ImageDraw.Draw(im, 'RGBA')

    if len(fixations[0]) == 3:
        x0, y0, duration = fixations[0]
    else:
        x0, y0 = fixations[0]

    for index, fixation in enumerate(fixations):
        
        if len(fixations[0]) == 3:
            duration = fixation[2]
            if 5 * (duration / 100) < 5:
                r = 3
            else:
                 r = 5 * (duration / 100)
        else:
            r = 8

        x = fixation[0]
        y = fixation[1]

        bound = (x - r, y - r, x + r, y + r)
        outline_color = (50, 255, 0, 0)
        
        if match_list[index] == 1:
        	fill_color = (50, 255, 0, 220)
        else:
        	fill_color = (255, 55, 0, 220)

        draw.ellipse(bound, fill=fill_color, outline=outline_color)

        bound = (x0, y0, x, y)
        line_color = (255, 155, 0, 155)
        penwidth = 2
        draw.line(bound, fill=line_color, width=5)

        # text_bound = (x + random.randint(-10, 10), y + random.randint(-10, 10))
        # text_color = (0, 0, 0, 225)
        # font = ImageFont.truetype("arial.ttf", 20)
        # draw.text(text_bound, str(index), fill=text_color,font=font)

        x0, y0 = x, y

    plt.figure(figsize=(17, 15))
    plt.imshow(np.asarray(im), interpolation='nearest')


def find_lines_Y(aois):
    ''' returns a list of line Ys '''
    
    results = []
    
    for index, row in aois.iterrows():
        y, height = row['y'], row['height']
        
        if y + height / 2 not in results:
            results.append(y + height / 2)
            
    return results



def find_word_centers(aois):
    ''' returns a list of word centers '''
    
    results = []
    
    for index, row in aois.iterrows():
        x, y, height, width = row['x'], row['y'], row['height'], row['width']
        
        center = [int(x + width // 2), int(y + height // 2)]
        
        if center not in results:
            results.append(center)
            
    return results


def find_word_centers_and_duration(aois):
    ''' returns a list of word centers '''
    
    results = []
    
    for index, row in aois.iterrows():
        x, y, height, width, token = row['x'], row['y'], row['height'], row['width'], row['token']
        
        center = [int(x + width // 2), int(y + height // 2), len(token) * 50]

        if center not in results:
            results.append(center)
    
    #print(results)
    return results



def overlap(fix, AOI):
    """checks if fixation is within AOI"""
    
    box_x = AOI.x
    box_y = AOI.y
    box_w = AOI.width
    box_h = AOI.height

    if fix[0] >= box_x and fix[0] <= box_x + box_w \
    and fix[1] >= box_y and fix[1] <= box_y + box_h:
        return True
    
    else:
        
        return False
    
    
def correction_quality(aois, original_fixations, corrected_fixations):
    
    match = 0
    total_fixations = len(original_fixations)
    results = [0] * total_fixations
    
    for index, fix in enumerate(original_fixations):
        
        for _, row  in aois.iterrows():
            
            if overlap(fix, row) and overlap(corrected_fixations[index], row):
                match += 1
                results[index] = 1
                
    return match / total_fixations, results