import tkinter as tk
import numpy as np
import PIL
from PIL import Image, ImageDraw
import tensorflow as tf


class App(tk.Tk):
	
	def __init__(self):
		
		super().__init__()
		
		# Canvas to draw image
		self.canvas = tk.Canvas(self, width = 280, height = 280, bg = "black")
		self.canvas.bind("<B1-Motion>", self.draw)
		self.canvas.pack(side='left', padx=10, pady=10)
		
		# Predict label
		self.label = tk.Label(self, font=("Comic Sans MS", 50))
		self.label.pack(pady=50)
		
		# Get image from canvas
		self.image = Image.new("RGB", (280, 280), (0, 0, 0))
		self.draw = ImageDraw.Draw(self.image)
		
		
		# Clear button
		self.clear_butt = tk.Button(self, text='Clear', width = 20, command = self.clear)
		self.clear_butt.pack(side='bottom', padx=10, pady=10)
		
		# Get digit button
		self.get_butt = tk.Button(self, text='Get digit', width = 20, command = self.get)
		self.get_butt.pack(side='bottom', padx=10)
		
		#Our model
		self.model = tf.keras.models.load_model('model.h5')
	
	def clear(self):
	
		self.canvas.delete("all")
		self.draw.rectangle((0, 0, 280, 280), fill=(0, 0, 0))
		
		
	def draw(self, event):
	
		x = event.x
		y = event.y 
	
		self.canvas.create_oval(x-10, y-10, x+10, y+10, fill='white', outline="")
		
		self.draw.ellipse((x-10, y-10, x+10, y+10), fill = 'white', outline =None)
		
	def get(self):
		
		img=self.image.convert("L").resize((28, 28), resample=Image.BILINEAR)
		
		
		data=np.array(img).reshape(1, 28, 28)
		data = data/255
		
		digit = self.model.predict_classes(data)[0]
		
		self.label.configure(text= str(digit))
		
def main():
	app = App()

	app.title('Handwritten digit recognition GUI') 

	app.mainloop()
	
	
if __name__ == '__main__':
	
	main()