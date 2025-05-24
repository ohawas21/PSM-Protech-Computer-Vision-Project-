from roboflow import Roboflow

rf      = Roboflow(api_key="FwIoqM4M31fvU1lcB1bI")
project = rf.workspace("pse-etxye").project("pse-mp46x")
version = project.version(4)

dataset = version.download(
    "yolov11",
    size="original",
    include_augmented=False
)

print("Downloaded to:", dataset.location)
