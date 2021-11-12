import pygame
import math
from random import randint
from sklearn.cluster import KMeans

def distance(p1, p2):
	return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2);

pygame.init()

screen = pygame.display.set_mode((800, 500))

pygame.display.set_caption("kmeans visualization")

running = True

clock = pygame.time.Clock()

# Colors
BACKGROUND = (214, 214, 214)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BACKGROUND_PANEL = (249, 255, 230)
RED = (250, 70, 90)
GREEN = (100, 150, 120)
BLUE = (53, 88, 171)
YELLOW = (209, 209, 29)
PURPLE = (166, 26, 217)
SKY = (31, 222, 219)
ORANGE = (222, 133, 31)
PINK = (214, 24, 163)
DARK_GREEN = (9, 46, 10)
COLORS = [RED, GREEN, BLUE, YELLOW, PURPLE, SKY, ORANGE, PINK, DARK_GREEN]

# Fonts
font = pygame.font.SysFont('sans', 40)
font_small = pygame.font.SysFont('sans', 20)

# Text render
text_plus = font.render('+', True, WHITE)
text_minus = font.render('-', True, WHITE)
text_run = font.render('RUN', True, WHITE)
text_random = font.render('RANDOM', True, WHITE)
text_algorithm = font.render('ALGORITHM', True, WHITE)
text_reset = font.render('RESET', True, WHITE)

K = 0
err = 0
points = []
clusters = []
labels = []

while running: 
	clock.tick(60)
	screen.fill(BACKGROUND)
	mouse_x, mouse_y = pygame.mouse.get_pos()

	# Draw interface

	# Draw panel
	pygame.draw.rect(screen, BLACK, (50, 50, 350, 400))
	pygame.draw.rect(screen, BACKGROUND_PANEL, (55, 55, 340, 390))

	# K button
	pygame.draw.rect(screen, BLACK, (500, 50, 40, 40))
	screen.blit(text_plus, (507, 50))

	pygame.draw.rect(screen, BLACK, (600, 50, 40, 40))
	screen.blit(text_minus, (613, 43))

	# K value
	text_k = font.render("K = " + str(K), True, BLACK)
	screen.blit(text_k, (670, 45))

	# Run button 
	pygame.draw.rect(screen, BLACK, (500, 125, 100, 40))
	screen.blit(text_run, (505, 123))

	# Random button
	pygame.draw.rect(screen, BLACK, (500, 200, 185, 40))
	screen.blit(text_random, (505, 200))

	# Algorithm button
	pygame.draw.rect(screen, BLACK, (500, 275, 245, 40))
	screen.blit(text_algorithm, (505, 275))

	# Reset button
	pygame.draw.rect(screen, BLACK, (500, 350, 145, 40))
	screen.blit(text_reset, (505, 350))

	# Mouse position in the panel
	if 55 < mouse_x < 395 and 55 < mouse_y < 445:
		text_mouse = font_small.render("(" + str(mouse_x - 55) + ", " + str(mouse_y - 55) + ")", True, BLACK)
		screen.blit(text_mouse, (mouse_x + 10, mouse_y))
	# End drawing interface

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False
		if event.type == pygame.MOUSEBUTTONDOWN:
			# Create points on panel
			if 55 < mouse_x < 395 and 55 < mouse_y < 445:
				labels = []
				point = [mouse_x - 55, mouse_y - 55]
				points.append(point)
			# Change K button
			if 500 < mouse_x < 540 and 50 < mouse_y < 90:
				if K < 9:
					K += 1
			if 600 < mouse_x < 640 and 50 < mouse_y < 90:
				if K > 0:
					K -= 1
			# Run button
			if 500 < mouse_x < 600 and 125 < mouse_y < 165:
				labels = []

				if clusters != []:
					# Label the points
					for point in points:
						distance_to_clusters = []
						for cluster in clusters:
							distance_to_clusters.append(distance(point, cluster))
							
						min_distance = min(distance_to_clusters)
						label = distance_to_clusters.index(min_distance)
						labels.append(label)

					# Center the clusters
					for i in range(len(clusters)):
						sum_x = 0;
						sum_y = 0;
						count = 0;
						for j in range(len(points)):
							if labels[j] == i:
								sum_x += points[j][0]
								sum_y += points[j][1]
								count += 1

						if count != 0:
							new_x = sum_x / count
							new_y = sum_y / count
							clusters[i] = [new_x, new_y]

			# Random button
			if 500 < mouse_x < 685 and 200 < mouse_y < 240:
				clusters = []
				labels = []
				for i in range(K):
					cluster = [randint(0, 340), randint(0, 390)]
					clusters.append(cluster)
			# Algorithm button
			if 500 < mouse_x < 745 and 275 < mouse_y < 315:
				if K > 0 and points != []:
					kmeans = KMeans(n_clusters=K).fit(points)
					clusters = kmeans.cluster_centers_
					labels = kmeans.labels_

			# Reset button
			if 500 < mouse_x < 645 and 350 < mouse_y < 390:
				labels = []
				points = []
				clusters = []
				K = 0
				err = 0

	# Draw panel points
	for i in range(len(points)):
		pygame.draw.circle(screen, BLACK, (points[i][0] + 55, points[i][1] + 55), 5)

		if labels == []:
			pygame.draw.circle(screen, WHITE, (points[i][0] + 55, points[i][1] + 55), 4)
		else:
			pygame.draw.circle(screen, COLORS[labels[i]], (points[i][0] + 55, points[i][1] + 55), 4)
			
	# Draw clusters
	for i in range(len(clusters)):
		pygame.draw.circle(screen, COLORS[i], (int(clusters[i][0]) + 55, int(clusters[i][1]) + 55), 7)

	# Calculate and draw error
	err = 0
	if clusters != [] and labels != []:
		for i in range(len(points)):
			err += distance(points[i], clusters[labels[i]])

	text_error = font.render("Error: " + str(int(err)), True, BLACK)
	screen.blit(text_error, (500, 430))

	pygame.display.flip()

pygame.quit()