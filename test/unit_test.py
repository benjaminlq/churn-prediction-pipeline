# %%pytest
import configs
import unittest
import tempfile
import subprocess
from kfp.components._components import _resolve_command_line_and_paths
import kfp.components as comp
from contextlib import contextmanager
from os import path

def components_local_output_dir_context(output_dir: str):
    old_dir = comp._components._outputs_dir
    try:
        comp._components._outputs_dir = output_dir
        yield output_dir
    finally:
        comp._components._outputs_dir = old_dir
        
class TestNotebook(unittest.TestCase):
    # check if {file_name} exsist in gcs {bucket_name}
    def checkFileExistInGCP(self, file_name, bucket_name):
        from google.cloud import storage
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        stats = storage.Blob(bucket=bucket, name=file_name).exists(storage_client)
        self.assertEqual(stats, True)

    def test_data_ingest(self):
        expected = 'assignment2/data/data_banknote_authentication.txt'
        bucket = BUCKET_NAME
        actual = get_data_func(url = DATA_PATH, bucket_name = bucket)
        self.assertEqual(actual, expected)
        self.checkFileExistInGCP(expected, bucket)

    def test_xgboost_model_locally(self):
        data_path = 'assignment2/data/data_banknote_authentication.txt'
        bucket = BUCKET_NAME
        model_name = "model"
        with tempfile.TemporaryDirectory() as temp_dir_name:
            with components_local_output_dir_context(temp_dir_name):
                task = train_op(data_path = data_path,
                      bucket = bucket,
                      model_name = model_name)
                resolved_cmd = _resolve_command_line_and_paths(
                    task.component_ref.spec,
                    task.arguments,
                )

            full_command = resolved_cmd.command + resolved_cmd.args
            subprocess.run(["pip", "install", "xgboost"])
            subprocess.run(full_command, check=True)
        out_path = "model.bst"
        assert path.exists(out_path) is True
        self.checkFileExistInGCP(f'assignment2/{model_name}/{model_name}.bst', bucket)
        
    ## Test deploy component
    ## Correct input and output
    def test_endpoint_created(self):
        from kfp.components.structures import InputSpec, OutputSpec
        model_uri = MODEL_URI
        print(model_uri)
        project = PROJECT_ID
        region =REGION
        serving_container_image_uri  = SERVING_IMAGE_URI
        with tempfile.TemporaryDirectory() as temp_dir_name:
            with components_local_output_dir_context(temp_dir_name):
                task = deploy_op(
                model_uri = model_uri,
                project=project,
                region=region, 
                serving_container_image_uri = serving_container_image_uri,
                )
                self.assertEqual(
                    task.component_ref.spec.inputs,
                [InputSpec(name='model_uri', type='String', description=None, default=None, optional=False, annotations=None),
                InputSpec(name='project', type='String', description=None, default=None, optional=False, annotations=None), 
                InputSpec(name='region', type='String', description=None, default=None, optional=False, annotations=None),
                InputSpec(name='serving_container_image_uri', type='String', description=None, default=None, optional=False, annotations=None)])
                self.assertEqual(task.component_ref.spec.outputs, [OutputSpec(name='vertex_model', type='Model', description=None, annotations=None)])

unittest.main(argv=[''], verbosity=1, exit=False)