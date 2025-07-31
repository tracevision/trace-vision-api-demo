import argparse
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import pandas as pd
import cv2
import numpy as np
import json


def ll_to_pixel(lat, lon, map_metadata):
    """Converts lat/lon to pixel coordinates on the map."""
    center_lat = map_metadata["center_lat"]
    center_lon = map_metadata["center_lon"]
    side_length_m = map_metadata["side_length_m"]
    side_length_px = map_metadata["side_length_px"]

    # Simple equirectangular projection (approximation)
    # This is less accurate than more complex projections but good for small areas.
    m_per_deg_lat = 111132.954 - 559.822 * np.cos(2 * np.radians(center_lat)) + 1.175 * np.cos(4 * np.radians(center_lat))
    m_per_deg_lon = 111319.488 * np.cos(np.radians(center_lat))

    dx_m = (lon - center_lon) * m_per_deg_lon
    dy_m = (lat - center_lat) * m_per_deg_lat

    pixels_per_meter = side_length_px / side_length_m

    x_px = int((side_length_px / 2) + dx_m * pixels_per_meter)
    y_px = int((side_length_px / 2) - dy_m * pixels_per_meter) # y is inverted in image coordinates

    return x_px, y_px


class VideoPlayerWindow(tk.Toplevel):
    def __init__(self, parent, resources_path, object_id1, object_id2):
        super().__init__(parent)
        self.title(f"Video Comparison: {object_id1} vs {object_id2}")
        self.parent = parent
        
        self.video_path1 = os.path.join(resources_path, object_id1, "video.mp4")
        self.video_path2 = os.path.join(resources_path, object_id2, "video.mp4")

        self.cap1 = cv2.VideoCapture(self.video_path1)
        self.cap2 = cv2.VideoCapture(self.video_path2)

        # Main frame
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # --- Left side: Video Player ---
        video_frame = ttk.Frame(main_frame)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        instruction_label = ttk.Label(video_frame, text="Videos are playing on a loop. Close this window to return.")
        instruction_label.pack(pady=5)
        
        self.video_label = tk.Label(video_frame)
        self.video_label.pack()

        # --- Right side: Map and Metadata ---
        info_frame = ttk.Frame(main_frame, padding="10")
        info_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # Load map and metadata
        map_path = os.path.join(resources_path, "satellite_map.png")
        map_metadata_path = os.path.join(resources_path, "map_metadata.json")
        
        if os.path.exists(map_path) and os.path.exists(map_metadata_path):
            with open(map_metadata_path, 'r') as f:
                map_metadata = json.load(f)

            # Load track metadata
            metadata1 = self.load_object_metadata(resources_path, object_id1)
            metadata2 = self.load_object_metadata(resources_path, object_id2)

            map_image_pil = Image.open(map_path)
            
            # Draw on map
            if metadata1 and metadata2:
                map_image_cv = cv2.cvtColor(np.array(map_image_pil), cv2.COLOR_RGB2BGR)
                
                # Track 1
                px1, py1 = ll_to_pixel(metadata1['mean_lat'], metadata1['mean_lon'], map_metadata)
                cv2.circle(map_image_cv, (px1, py1), 5, (0, 255, 0), -1) # Green dot
                cv2.putText(map_image_cv, f"1: {object_id1}", (px1 + 10, py1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Track 2
                px2, py2 = ll_to_pixel(metadata2['mean_lat'], metadata2['mean_lon'], map_metadata)
                cv2.circle(map_image_cv, (px2, py2), 5, (0, 0, 255), -1) # Red dot
                cv2.putText(map_image_cv, f"2: {object_id2}", (px2 + 10, py2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                map_image_pil = Image.fromarray(cv2.cvtColor(map_image_cv, cv2.COLOR_BGR2RGB))
            
            self.map_imgtk = ImageTk.PhotoImage(image=map_image_pil)
            map_label = ttk.Label(info_frame, image=self.map_imgtk)
            map_label.pack(pady=10)
            
            # Display metadata text
            if metadata1:
                ttk.Label(info_frame, text=f"Object 1 ({object_id1}):", font=("", 10, "bold")).pack(anchor='w')
                ttk.Label(info_frame, text=f"  Start: {metadata1['start_time']}").pack(anchor='w')
                ttk.Label(info_frame, text=f"  End: {metadata1['end_time']}").pack(anchor='w')
            if metadata2:
                ttk.Label(info_frame, text=f"Object 2 ({object_id2}):", font=("", 10, "bold")).pack(anchor='w', pady=(10, 0))
                ttk.Label(info_frame, text=f"  Start: {metadata2['start_time']}").pack(anchor='w')
                ttk.Label(info_frame, text=f"  End: {metadata2['end_time']}").pack(anchor='w')

        self.transient(parent)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.bind("<Map>", self._on_map)
        
        self.update_video()

    def load_object_metadata(self, resources_path, object_id):
        metadata_path = os.path.join(resources_path, object_id, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return None

    def _on_map(self, event):
        self.grab_set()
        self.unbind("<Map>")

    def update_video(self):
        ret1, frame1 = self.cap1.read()
        if not ret1:
            self.cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret1, frame1 = self.cap1.read()

        ret2, frame2 = self.cap2.read()
        if not ret2:
            self.cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret2, frame2 = self.cap2.read()

        if ret1 and ret2:
            combined_frame = np.hstack((frame1, frame2))
            img = Image.fromarray(cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.config(image=imgtk)
            self.video_label.image = imgtk
            self.after(33, self.update_video)

    def on_close(self):
        self.cap1.release()
        self.cap2.release()
        self.destroy()


class ComparisonWindow(tk.Toplevel):
    def __init__(self, parent, resources_path, selected_object_id, similarity_threshold=0.0):
        super().__init__(parent)
        self.title(f"Top Matches for {selected_object_id}")
        self.parent = parent
        self.resources_path = resources_path
        self.selected_object_id = selected_object_id
        self.similarity_threshold = similarity_threshold
        self.geometry("1400x850")

        self.df_filtered = None
        self.thumbnail_widgets = []
        self.current_thumb_index = 0
        self.batch_size = 50
        self.loading = False

        self.loading_label = ttk.Label(self, text="Loading...")

        instruction_label = ttk.Label(self, text=f"Showing matches for {self.selected_object_id}. Click a thumbnail to compare videos.")
        instruction_label.pack(pady=5)

        self.show_same_camera = tk.BooleanVar(value=False)
        cb = ttk.Checkbutton(
            self,
            text="Include matches from the same camera",
            variable=self.show_same_camera,
            command=self.reset_and_load_thumbnails,
        )
        cb.pack(pady=5)

        reference_frame = ttk.Frame(self, padding="10")
        reference_frame.pack()
        
        ref_thumb_path = os.path.join(self.resources_path, self.selected_object_id, "thumbnail.jpg")
        if os.path.exists(ref_thumb_path):
            ref_img = Image.open(ref_thumb_path)
            ref_img.thumbnail((128, 256))
            self.ref_imgtk = ImageTk.PhotoImage(image=ref_img)

            ref_thumb_label = ttk.Label(reference_frame, image=self.ref_imgtk)
            ref_thumb_label.pack()

            ref_text_label = ttk.Label(reference_frame, text="Reference Object")
            ref_text_label.pack()

        self.canvas = tk.Canvas(self)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self._on_scroll)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        self.canvas.bind("<Configure>", self.reflow_thumbnails)
        self.bind("<MouseWheel>", self._on_mousewheel)
        self.bind("<Button-4>", self._on_mousewheel)
        self.bind("<Button-5>", self._on_mousewheel)

        self.transient(parent)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.bind("<Map>", self._on_map)

    def _on_map(self, event):
        self.grab_set()
        self.update_idletasks()
        self.after(50, self.reset_and_load_thumbnails)
        self.unbind("<Map>")

    def _on_scroll(self, *args):
        self.scrollbar.set(*args)
        if float(args[1]) > 0.9 and not self.loading:
            self.load_more_similar_thumbnails()

    def _on_mousewheel(self, event):
        if hasattr(event, 'num') and event.num in (4, 5):
            if event.num == 4:
                self.canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self.canvas.yview_scroll(1, "units")
        elif hasattr(event, 'delta'):
            if event.delta > 0:
                self.canvas.yview_scroll(-1, "units")
            elif event.delta < 0:
                self.canvas.yview_scroll(1, "units")

    def load_similarity_data(self):
        csv_path = os.path.join(self.resources_path, self.selected_object_id, "similarities.csv")
        df = pd.read_csv(csv_path)

        selected_camera_id = self.selected_object_id.split('_')[0]
        include_same_camera = self.show_same_camera.get()

        filtered_rows = []
        for _, row in df.iterrows():
            object_id = str(row["object_id"])
            other_camera_id = object_id.split('_')[0]
            similarity = row["cosine_similarity"]

            if similarity < self.similarity_threshold:
                continue

            if include_same_camera or (other_camera_id != selected_camera_id):
                filtered_rows.append(row)
        
        if not filtered_rows:
            self.df_filtered = pd.DataFrame()
            no_matches_label = ttk.Label(self.scrollable_frame, text="No matches found for this filter setting.")
            no_matches_label.pack()
            return

        self.df_filtered = pd.DataFrame(filtered_rows)

    def reset_and_load_thumbnails(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        self.thumbnail_widgets = []
        self.current_thumb_index = 0
        
        self.load_similarity_data()
        self.load_more_similar_thumbnails()

    def load_more_similar_thumbnails(self):
        if self.loading or self.df_filtered is None or self.current_thumb_index >= len(self.df_filtered):
            return
        
        self.loading = True
        self.loading_label.pack(pady=10)
        self.update_idletasks()

        end_index = min(self.current_thumb_index + self.batch_size, len(self.df_filtered))

        for index in range(self.current_thumb_index, end_index):
            row = self.df_filtered.iloc[index]
            object_id = row.object_id
            similarity = row.cosine_similarity

            thumb_path = os.path.join(self.resources_path, str(object_id), "thumbnail.jpg")
            if os.path.exists(thumb_path):
                img = Image.open(thumb_path)
                img.thumbnail((128, 256))
                imgtk = ImageTk.PhotoImage(image=img)

                button = ttk.Button(
                    self.scrollable_frame,
                    text=f"Sim: {similarity:.3f}",
                    image=imgtk,
                    compound="top",
                    command=lambda oid=object_id: self.on_thumbnail_click(oid),
                    style="Thumbnail.TButton",
                )
                button.image = imgtk
                self.thumbnail_widgets.append(button)
        
        self.current_thumb_index = end_index
        self.reflow_thumbnails()
        self.update_idletasks()
        self.loading = False
        self.loading_label.pack_forget()

    def reflow_thumbnails(self, event=None):
        canvas_width = self.canvas.winfo_width()
        if canvas_width <= 1 or not self.thumbnail_widgets:
            return

        thumb_width = self.thumbnail_widgets[0].winfo_reqwidth() + 10
        num_cols = max(1, canvas_width // thumb_width)

        for i, widget in enumerate(self.thumbnail_widgets):
            row = i // num_cols
            col = i % num_cols
            widget.grid(row=row, column=col, padx=5, pady=5)

        self.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_thumbnail_click(self, other_object_id):
        VideoPlayerWindow(self, self.resources_path, self.selected_object_id, str(other_object_id))
        
    def on_close(self):
        self.destroy()


class App(tk.Tk):
    def __init__(self, resources_path):
        super().__init__()
        self.title("All Objects - Thumbnail Viewer")
        self.geometry("1400x850")
        self.resources_path = resources_path
        self.object_ids = []
        self.all_object_ids = []
        self.thumbnail_widgets = []
        self.current_object_index = 0
        self.batch_size = 50
        self.loading = False
        self.similarity_threshold = tk.DoubleVar(value=0.0)

        self.style = ttk.Style(self)
        self.style.configure("Thumbnail.TButton", padding=5)
        self.style.map(
            "Thumbnail.TButton",
            background=[("active", "#e0e0e0")],
            relief=[("hover", "raised")],
        )

        self.loading_label = ttk.Label(self, text="Loading...")

        instruction_label = ttk.Label(self, text="Click on a thumbnail to see its most similar matches.")
        instruction_label.pack(pady=5)

        # Frame for threshold controls
        threshold_frame = ttk.Frame(self, padding="5")
        threshold_frame.pack(fill='x', expand=False)

        threshold_label = ttk.Label(threshold_frame, text="Similarity Threshold:")
        threshold_label.pack(side='left', padx=(5, 2))
        
        threshold_entry = ttk.Entry(threshold_frame, textvariable=self.similarity_threshold, width=10)
        threshold_entry.pack(side='left', padx=(0, 5))

        apply_button = ttk.Button(threshold_frame, text="Apply Filter", command=self.apply_filter)
        apply_button.pack(side='left')

        self.canvas = tk.Canvas(self)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self._on_scroll)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.canvas.bind("<Configure>", self.reflow_thumbnails)
        self.bind("<MouseWheel>", self._on_mousewheel)
        self.bind("<Button-4>", self._on_mousewheel)
        self.bind("<Button-5>", self._on_mousewheel)

        self.load_object_ids()
        self.apply_filter()

    def _on_scroll(self, *args):
        self.scrollbar.set(*args)
        if float(args[1]) > 0.9 and not self.loading:
            self.load_more_thumbnails()

    def _on_mousewheel(self, event):
        if hasattr(event, 'num') and event.num in (4, 5):
            if event.num == 4:
                self.canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self.canvas.yview_scroll(1, "units")
        elif hasattr(event, 'delta'):
            if event.delta > 0:
                self.canvas.yview_scroll(-1, "units")
            elif event.delta < 0:
                self.canvas.yview_scroll(1, "units")

    def load_object_ids(self):
        self.all_object_ids = [d for d in os.listdir(self.resources_path) if os.path.isdir(os.path.join(self.resources_path, d))]

    def apply_filter(self):
        threshold = self.similarity_threshold.get()
        self.object_ids = []
        for object_id in self.all_object_ids:
            csv_path = os.path.join(self.resources_path, object_id, "similarities.csv")
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
            if df.empty:
                continue

            # Check if there are any matches above the threshold
            if (df['cosine_similarity'] >= threshold).any():
                self.object_ids.append(object_id)

        # Reset and reload thumbnails
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.thumbnail_widgets = []
        self.current_object_index = 0
        self.load_more_thumbnails()


    def load_more_thumbnails(self):
        if self.loading or self.current_object_index >= len(self.object_ids):
            return
        self.loading = True
        self.loading_label.pack(pady=10)
        self.update_idletasks()

        end_index = min(self.current_object_index + self.batch_size, len(self.object_ids))

        for i in range(self.current_object_index, end_index):
            object_id = self.object_ids[i]
            thumb_path = os.path.join(self.resources_path, object_id, "thumbnail.jpg")
            if os.path.exists(thumb_path):
                img = Image.open(thumb_path)
                img.thumbnail((128, 256))
                imgtk = ImageTk.PhotoImage(image=img)

                button = ttk.Button(
                    self.scrollable_frame,
                    text=object_id,
                    image=imgtk,
                    compound="top",
                    command=lambda oid=object_id: self.on_thumbnail_click(oid),
                    style="Thumbnail.TButton",
                )
                button.image = imgtk
                self.thumbnail_widgets.append(button)
        
        self.current_object_index = end_index
        self.reflow_thumbnails()
        self.update_idletasks()
        self.loading = False
        self.loading_label.pack_forget()

    def reflow_thumbnails(self, event=None):
        canvas_width = self.canvas.winfo_width()
        if canvas_width <= 1 or not self.thumbnail_widgets:
            return

        thumb_width = self.thumbnail_widgets[0].winfo_reqwidth() + 10
        num_cols = max(1, canvas_width // thumb_width)

        for i, widget in enumerate(self.thumbnail_widgets):
            row = i // num_cols
            col = i % num_cols
            widget.grid(row=row, column=col, padx=5, pady=5)
    
        self.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_thumbnail_click(self, object_id):
        ComparisonWindow(self, self.resources_path, object_id, self.similarity_threshold.get())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path_to_resources", required=True, help="Path to the tkinter_resources directory")
    args = ap.parse_args()

    app = App(args.path_to_resources)
    app.mainloop()


if __name__ == "__main__":
    main()
