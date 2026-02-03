#!/usr/bin/env python3

"""Visual status display for policy nodes."""

import sys
import time


class StatusDisplay:
	"""Visual status bar showing observation freshness and queue timeline."""
	
	def __init__(self, enable: bool = True):
		self.enable = enable
		self._last_update = 0.0
		self._update_interval = 0.05  # Update every 50ms max
		self._reserved_lines = 0
		if self.enable:
			# Start with minimal reservation, will expand as needed
			self._reserved_lines = 10
			print("\n" * self._reserved_lines, end='', flush=True)
	
	def update(self, obs_status: dict, queue_info: dict):
		if not self.enable:
			return
		now = time.monotonic()
		if now - self._last_update < self._update_interval:
			return
		self._last_update = now
		
		# ANSI codes
		SAVE_POS = "\033[s"
		RESTORE_POS = "\033[u"
		CLEAR_LINE = "\033[2K"
		GREEN = "\033[92m"
		RED = "\033[91m"
		YELLOW = "\033[93m"
		BLUE = "\033[94m"
		CYAN = "\033[96m"
		MAGENTA = "\033[95m"
		RESET = "\033[0m"
		BOLD = "\033[1m"
		DIM = "\033[2m"
		
		# Build observation status line (all on one line)
		obs_items = [
			('STATE', obs_status.get('state', False)),
			('GRIP', obs_status.get('gripper', False)),
			('FRONT', obs_status.get('front', False)),
			('DOWN', obs_status.get('down', False)),
			('3POV', obs_status.get('third', False)),
		]
		obs_parts = []
		for name, is_new in obs_items:
			color = GREEN if is_new else DIM + RED
			status = "●" if is_new else "○"
			obs_parts.append(f"{color}{status}{name}{RESET}")
		obs_line = "OBS: " + " ".join(obs_parts)
		
		# Build queue info line
		queue_size = queue_info.get('size', 0)
		queue_max = queue_info.get('max', 64)
		chunks = queue_info.get('chunks', [])
		
		queue_header = f"QUEUE: {queue_size}/{queue_max} chunks={len(chunks)}"
		
		# Build chunk timeline visualization (all chunks with shared time axis)
		timeline_lines = []
		if len(chunks) > 0:
			now_time = time.monotonic()
			timeline_width = 60
			
			# Find global time range across ALL chunks
			all_ts = [item['ts'] for chunk in chunks for item in chunk if len(chunk) > 0]
			
			if all_ts:
				global_min_ts = min(all_ts)
				global_max_ts = max(all_ts)
				time_span = max(0.5, global_max_ts - global_min_ts + 0.1)
				
				# Different colors for different chunks
				chunk_colors = [BLUE, CYAN, MAGENTA, YELLOW, GREEN]
				
				# Display all chunks
				for chunk_idx, chunk in enumerate(chunks):
					if len(chunk) == 0:
						continue
						
					timeline = [' '] * timeline_width
					color = chunk_colors[chunk_idx % len(chunk_colors)]
					
					# Mark actions in this chunk relative to global timeline
					for item in chunk:
						ts = item['ts']
						pos = int(((ts - global_min_ts) / time_span) * (timeline_width - 1))
						pos = max(0, min(timeline_width - 1, pos))
						
						if ts < now_time:
							timeline[pos] = RED + '●' + RESET  # Late
						elif ts - now_time < 0.05:
							timeline[pos] = YELLOW + '●' + RESET  # Soon
						else:
							timeline[pos] = color + '●' + RESET  # Future
					
					# Mark current time on each line
					now_pos = int(((now_time - global_min_ts) / time_span) * (timeline_width - 1))
					if 0 <= now_pos < timeline_width and timeline[now_pos] == ' ':
						timeline[now_pos] = DIM + GREEN + '|' + RESET
					
					chunk_line = f"  C{chunk_idx}: [{''.join(timeline)}] {len(chunk)}a"
					timeline_lines.append(chunk_line)
				
				# Show time span on last line
				if timeline_lines:
					timeline_lines[-1] += f" span={time_span*1000:.0f}ms"
		
		# Calculate total lines needed
		total_lines = 2 + len(timeline_lines)  # obs + queue header + chunk lines
		
		# Expand reserved space if needed
		if total_lines > self._reserved_lines:
			extra_lines = total_lines - self._reserved_lines
			print("\n" * extra_lines, end='', flush=True)
			self._reserved_lines = total_lines
		
		# Move up the correct number of lines
		MOVE_UP = f"\033[{total_lines}A"
		
		# Print status display (save position, move up, print, restore)
		output_parts = [f"{SAVE_POS}{MOVE_UP}"]
		output_parts.append(f"{CLEAR_LINE}{BOLD}{obs_line}{RESET}\n")
		output_parts.append(f"{CLEAR_LINE}{BOLD}{queue_header}{RESET}\n")
		for line in timeline_lines:
			output_parts.append(f"{CLEAR_LINE}{line}\n")
		# Clear any extra lines if we have fewer chunks now
		for _ in range(self._reserved_lines - total_lines):
			output_parts.append(f"{CLEAR_LINE}\n")
		output_parts.append(RESTORE_POS)
		
		output = "".join(output_parts)
		sys.stdout.write(output)
		sys.stdout.flush()

