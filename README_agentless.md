# Agentless Integration

This project leverages code and data from the [Agentless](https://github.com/OpenAutoCoder/Agentless) project to establish verifiable reward signals for training our code localization model.

The Agentless dataset can be downloaded from the [v1.5.0 release](https://github.com/OpenAutoCoder/Agentless/releases/tag/v1.5.0).

The `agentless_localization.py` script provides utilities for inspecting and analyzing the JSONL data files.

Here's an example where we look at the results at the end of the Localization step from the SWE-Bench Lite dataset.

```bash
$ python agentless_localization.py --input_file agentless_swebench_lite/edit_location_samples/loc_outputs.jsonl --num_samples 2

Instance ID: django__django-10914

Found Files:
        django/core/files/storage.py
        django/conf/global_settings.py
        django/core/files/uploadhandler.py
        django/core/files/uploadedfile.py
        docs/conf.py
        django/core/files/temp.py

Found Related Locs:

        File Path: django/core/files/storage.py
        Related Locs:
                class: FileSystemStorage

        File Path: django/conf/global_settings.py
        Related Locs:
                variable: FILE_UPLOAD_PERMISSIONS

        File Path: django/core/files/uploadhandler.py
        Related Locs:


Found Edit Locs:

        Edit Location Set #1:
                File: django/core/files/storage.py
                        function: FileSystemStorage._save
                        line: 260
                File: django/conf/global_settings.py
                        line: 307
                File: django/core/files/uploadhandler.py
                        (no specific location)

        Edit Location Set #2:
                File: django/core/files/storage.py
                        function: FileSystemStorage._save
                        line: 260
                File: django/conf/global_settings.py
                        line: 307
                File: django/core/files/uploadhandler.py
                        (no specific location)

        Edit Location Set #3:
                File: django/conf/global_settings.py
                        line: 307
                File: django/core/files/storage.py
                        line: 260
                File: django/core/files/uploadhandler.py
                        (no specific location)

        Edit Location Set #4:
                File: django/core/files/storage.py
                        function: FileSystemStorage._save
                        line: 260
                File: django/conf/global_settings.py
                        line: 307
                File: django/core/files/uploadhandler.py
                        (no specific location)

================================================================================

Instance ID: astropy__astropy-14182

Found Files:
        astropy/io/ascii/rst.py
        astropy/io/ascii/ui.py
        astropy/io/ascii/core.py
        astropy/io/ascii/connect.py
        astropy/table/connect.py
        astropy/table/row.py
        astropy/table/table.py

Found Related Locs:

        File Path: astropy/io/ascii/rst.py
        Related Locs:
                class: RST
                class: SimpleRSTHeader
                class: SimpleRSTData

        File Path: astropy/io/ascii/ui.py
        Related Locs:
                function: get_writer
                function: write

        File Path: astropy/io/ascii/core.py
        Related Locs:
                function: _get_writer

Found Edit Locs:

        Edit Location Set #1:
                File: astropy/io/ascii/rst.py
                        class: RST
                        line: 60
                File: astropy/io/ascii/ui.py
                        function: get_writer
                        line: 864
                        function: write
                        line: 974
                File: astropy/io/ascii/core.py
                        function: _get_writer
                        line: 1814

        Edit Location Set #2:
                File: astropy/io/ascii/rst.py
                        class: RST
                        line: 60
                File: astropy/io/ascii/ui.py
                        function: get_writer
                        line: 864
                File: astropy/io/ascii/core.py
                        function: _get_writer
                        line: 1814

        Edit Location Set #3:
                File: astropy/io/ascii/core.py
                        line: 1814
                File: astropy/io/ascii/rst.py
                        class: RST
                        function: __init__
                        line: 60
                File: astropy/io/ascii/ui.py
                        (no specific location)

        Edit Location Set #4:
                File: astropy/io/ascii/rst.py
                        class: RST
                        line: 60
                File: astropy/io/ascii/core.py
                        function: _get_writer
                        line: 1814
                File: astropy/io/ascii/ui.py
                        function: get_writer
                        line: 901
```
