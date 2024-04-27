import h5py
import os
import torch


class SimulationDataHandler:
    def __init__(self, simulation, filename=None, buffer_limit=100):
        """
        Initialize the data handler.
        
        :param simulation: The simulation object.
        :param filename: The name of the file where data will be saved.
        :param buffer_limit: The maximum number of snapshots or events to buffer before flushing.
        """
        self.simulation = simulation
        self.filename = filename
        self.buffer_limit = buffer_limit
        self.data = {
            "metadata": self._gather_metadata(),
            "snapshots": [],
            "events": []
        }
        if filename:
            self._handle_file_existence()

    def _handle_file_existence(self):
        if os.path.exists(self.filename):
            print(f"The file '{self.filename}' already exists.")
            response = input("Choose an action: 'overwrite', 'append', or 'cancel' [o/a/c]: ").strip().lower()
            if response == 'o' or response == 'overwrite':
                os.remove(self.filename)
                print("The existing file will be overwritten.")
            elif response == 'c' or response == 'cancel':
                raise SystemExit("Operation cancelled by user.")
            elif response == 'a' or response == 'append':
                print("Data will be appended to the existing file.")
            else:
                raise ValueError("Invalid input. Please restart and choose a valid action.")

    def _gather_metadata(self):
        return {
            "g": self.simulation.g,
            "v0": self.simulation.v0,
            "lattice_params": self.simulation.lattice.get_params()
        }

    def collect_snapshot(self):
        snapshot = (self.simulation.t, self.simulation.lattice.query_lattice_state())
        self.data["snapshots"].append(snapshot)
        if len(self.data["snapshots"]) >= self.buffer_limit:
            self.flush_data()

    def collect_event(self, event):
        if event:
            event_data = {
                "time": self.simulation.t,
                "event_type": event.etype.value,
                "x": event.x,
                "y": event.y,
            }
            self.data["events"].append(event_data)

    def flush_data(self):
        if self.filename:
            with h5py.File(self.filename, "a") as file:  # Using append mode
                self._export_snapshots(file)
                self._export_events(file)
            self.data["snapshots"] = []
            self.data["events"] = []

    def _export_snapshots(self, file):
        snapshots_group = file.require_group("snapshots")
        for time, snapshot in self.data["snapshots"]:
            if isinstance(snapshot, torch.Tensor):
                snapshot = snapshot.numpy()
            dataset = snapshots_group.create_dataset(f"snapshot_{len(snapshots_group)}", data=snapshot)
            dataset.attrs["time"] = time

    def _export_events(self, file):
        events_group = file.require_group("events")
        for event in self.data["events"]:
            event_dataset = events_group.create_dataset(f"event_{len(events_group)}", data=list(event.values()))
            for key in event.keys():
                event_dataset.attrs[key] = event[key]

    def close(self):
        if self.filename:
            self.flush_data()  # Ensure all data is written before closing


