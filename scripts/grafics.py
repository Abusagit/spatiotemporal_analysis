{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip3 install geopandas"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import geopandas as gpd\n",
    "from matplotlib import pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install folium"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "import folium\n",
    "from folium.plugins import HeatMap\n",
    "\n",
    "m = folium.Map(location=[34.15, -118.243683],\n",
    "               zoom_start=11,\n",
    "               tiles='cartodbpositron')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from pathlib import Path\n",
    "\n",
    "DATA_DIR = Path(\"../data/\")\n",
    "\n",
    "def read_dataset_file(path: Path) -> np.lib.npyio.NpzFile:\n",
    "    return np.load(file=path, allow_pickle=True)\n",
    "\n",
    "metr_la = read_dataset_file(DATA_DIR / \"metr_la_new.npz\")\n",
    "metr_la"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "metr_la[\"timestamp_frequency\"].item()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "EDGES_KEY = \"edges\"\n",
    "TARGETS_KEY = \"targets\"\n",
    "SPATIAL_FEATURES_KEY = \"spatial_node_features\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "metr_la[EDGES_KEY]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "metr_la_targets = metr_la[TARGETS_KEY]\n",
    "metr_la_targets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "timestamps = pd.date_range(start=metr_la[\"first_timestamp_datetime\"].item(),\n",
    "                           end=metr_la[\"last_timestamp_datetime\"].item(),\n",
    "                           freq=\"5min\",\n",
    "                           )\n",
    "timestamps"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# heat_data = [[r['pickup_latitude'], r['pickup_longitude']] for _, r in df.iterrows()]\n",
    "time = 0\n",
    "metr_la_coords = metr_la[SPATIAL_FEATURES_KEY]\n",
    "\n",
    "metr_la_coords.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.DataFrame(metr_la_coords[0], columns=['latitude', 'longitude'])\n",
    "\n",
    "# HeatMap(df, radius=10).add_to(m)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "m"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install osmnx"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install shapely"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "import shapely\n",
    "from shapely.geometry import Point, Polygon, LineString\n",
    "from shapely.geometry import MultiPoint, MultiLineString, MultiPolygon, GeometryCollection, LineString\n",
    "import json"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "multipoints = MultiPoint([[a, b] for a, b in metr_la_coords[0]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "multipoints"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "geojson = shapely.geometry.mapping(multipoints)\n",
    "geojson"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "colors = ['green', 'yellow', 'red']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install imageio\n",
    "!pip install selenium\n",
    "!pip install pillow\n",
    "!pip install html2image"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "from html2image import Html2Image\n",
    "\n",
    "hti = Html2Image()\n",
    "\n",
    "def get_timestamp_map(timestamp):\n",
    "    m = folium.Map(location=[34.15, -118.243683],\n",
    "                    zoom_start=11, \n",
    "                    tiles='cartodbpositron')\n",
    "    for i, point in enumerate(geojson['coordinates']):\n",
    "        normal_vel = (metr_la_targets[timestamp, i]) / 75\n",
    "        sensor_color = colors[0]\n",
    "        if normal_vel < 0.5 and normal_vel > 0.3:\n",
    "            sensor_color = colors[1]\n",
    "        elif normal_vel <= 0.3:\n",
    "            sensor_color = colors[2]\n",
    "        folium.CircleMarker(list(point), radius=5, fill=True, color=sensor_color, \n",
    "                            fill_color=sensor_color, fill_opacity=0.6).add_to(m)\n",
    "        \n",
    "    return m\n",
    "    \n",
    "    #m.save(f'map_{i}.html')\n",
    "    #hti.screenshot(url=f'map_{i}.html', save_as=f'./temp_frames/frame_{i}.png')\n",
    "\n",
    "get_timestamp_map(0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "start = 0\n",
    "count = 288 * 7"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Папка для временных файлов\n",
    "os.makedirs('temp_frames', exist_ok= True)\n",
    "\n",
    "# Генерируем несколько кадров с разным положением маркера\n",
    "for i in range(start, start + count):\n",
    "    timestep_map = get_timestamp_map(i)\n",
    "    timestep_map.save(f'temp_frames/map_{i}.html')\n",
    "    \n",
    "\"\"\"# Собираем GIF\n",
    "frames = [Image.open(f'./temp_frames/frame_{i}.png') for i in range(count)]\n",
    "frames[0].save('map_animation.gif', format='GIF', append_images=frames[1:],\n",
    "               save_all=True, duration=100, loop=0)\n",
    "hti.screenshot(url=f'map_{i}.html', save_as=f'/Users/macbook/Documents/projects/graphs/spatiotemporal_analysis/notebooks/temp_frames/frame_{i}.png')\n",
    "\n",
    "# Чистим временные файлы\n",
    "for i in range(count):\n",
    "    os.remove(f'temp_frames/frame_{i}.png')\n",
    "    os.remove(f'temp_frames/map_{i}.html')\n",
    "os.rmdir('temp_frames')\"\"\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "from shutil import copyfile\n",
    "\n",
    "def html_to_image(html_file, output):\n",
    "    path = Path(\"temp_frames/\" + output)\n",
    "    if path.exists():\n",
    "        return\n",
    "    \n",
    "    hti = Html2Image()\n",
    "\n",
    "    hti = Html2Image(\n",
    "        custom_flags=[\"--headless\", \"--disable-gpu\", \"--no-sandbox\"]\n",
    "    )\n",
    "    \n",
    "    hti.screenshot(url=html_file, save_as=output)\n",
    "    copyfile(output, \"temp_frames/\" + output)\n",
    "    os.remove(output)\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install tqdm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from tqdm import tqdm\n",
    "\n",
    "for i in tqdm(range(count)):\n",
    "    i += start \n",
    "    html_to_image(f'temp_frames/map_{i}.html', f'frame_{i}.png')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'tqdm' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[1], line 3\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;21;01mPIL\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;28;01mimport\u001b[39;00m Image\n\u001b[0;32m----> 3\u001b[0m frames \u001b[38;5;241m=\u001b[39m [Image\u001b[38;5;241m.\u001b[39mopen(\u001b[38;5;124mf\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mtemp_frames/frame_\u001b[39m\u001b[38;5;132;01m{\u001b[39;00mi\u001b[38;5;250m \u001b[39m\u001b[38;5;241m+\u001b[39m\u001b[38;5;250m \u001b[39mstart\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m.png\u001b[39m\u001b[38;5;124m'\u001b[39m) \u001b[38;5;28;01mfor\u001b[39;00m i \u001b[38;5;129;01min\u001b[39;00m \u001b[43mtqdm\u001b[49m(\u001b[38;5;28mrange\u001b[39m(\u001b[38;5;28mint\u001b[39m(\u001b[38;5;241m1\u001b[39m \u001b[38;5;241m*\u001b[39m count \u001b[38;5;241m/\u001b[39m \u001b[38;5;241m7\u001b[39m)))]\n\u001b[1;32m      4\u001b[0m \u001b[38;5;28;01mwith\u001b[39;00m tqdm(total\u001b[38;5;241m=\u001b[39m\u001b[38;5;28mlen\u001b[39m(frames), desc\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mСоздание анимации\u001b[39m\u001b[38;5;124m\"\u001b[39m) \u001b[38;5;28;01mas\u001b[39;00m pbar:\n\u001b[1;32m      5\u001b[0m     frames[\u001b[38;5;241m0\u001b[39m]\u001b[38;5;241m.\u001b[39msave(\n\u001b[1;32m      6\u001b[0m         \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mmap_animation.gif\u001b[39m\u001b[38;5;124m'\u001b[39m,\n\u001b[1;32m      7\u001b[0m         \u001b[38;5;28mformat\u001b[39m\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mGIF\u001b[39m\u001b[38;5;124m'\u001b[39m,\n\u001b[0;32m   (...)\u001b[0m\n\u001b[1;32m     13\u001b[0m         callback\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mlambda\u001b[39;00m _: pbar\u001b[38;5;241m.\u001b[39mupdate(\u001b[38;5;241m0\u001b[39m)\n\u001b[1;32m     14\u001b[0m     )\n",
      "\u001b[0;31mNameError\u001b[0m: name 'tqdm' is not defined"
     ]
    }
   ],
   "source": [
    "from PIL import Image\n",
    "\n",
    "frames = [Image.open(f'temp_frames/frame_{i + start}.png') for i in tqdm(range(int(1 * count / 7)))]\n",
    "with tqdm(total=len(frames), desc=\"Создание анимации\") as pbar:\n",
    "    frames[0].save(\n",
    "        'map_animation.gif',\n",
    "        format='GIF',\n",
    "        append_images=frames[1:],\n",
    "        save_all=True,\n",
    "        duration=100,\n",
    "        loop=0,\n",
    "        optimize=True,\n",
    "        callback=lambda _: pbar.update(0)\n",
    "    )\n",
    "for i in range(start, start + count):\n",
    "    os.remove(f'temp_frames/frame_{i}.png')\n",
    "    os.remove(f'temp_frames/map_{i}.html')\n",
    "os.rmdir('temp_frames')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "metr_la_edges = metr_la[EDGES_KEY]\n",
    "metr_la_edges"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "multiline = MultiLineString([LineString([Point(metr_la_coords[0][start]), Point(metr_la_coords[0][end])])\n",
    "                            for start, end in metr_la_edges])\n",
    "multiline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "geojson_line = shapely.geometry.mapping(multiline)\n",
    "geojson_line"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "for line, line2 in geojson_line['coordinates']:\n",
    "    print(line)\n",
    "    print(line2)\n",
    "    print(\"//\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 109,
   "metadata": {},
   "outputs": [],
   "source": [
    "for line in geojson_line['coordinates']:\n",
    "    folium.PolyLine([list(line[0]), list(line[1])], color='blue', weight=1, opacity=0.25).add_to(m)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "m"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": ".venv",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
