Guides
---------------------------------------------

Herein we have collected a selection of guides that tries to give a step-by-step solution to certain tasks that a
user of AutoAtom6000 might want to accomplish.

Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install AutoAtom6000, simply follow the steps below.

    1. Go to the **download** section of this webpage, and download the zip-file provided under **Executable** to a
    desired location on your computer.

    2. Navigate to the downloaded zip-file on your computer, right click and select 'extract all' or something similar,
    depending on your operating system or installed compression software.

    3. (Optional) Navigate into the extracted folder and locate the 'aacc.exe' -file. Right click and select 'Send to
    -> desktop (create shortcut)', if a desktop shortcut is desired.

.. note::

    When starting AutoAtom, there will be a significant waiting-time (\~20 sec) before the GUI loads. This is because
    the exe will first build its environment in temporary folders, which takes some time. Unfortunately, as a
    consequence of pyinstaller's --onefile option, during this time there will be no indication that the program is
    running, so be patient before clicking several times! In the future, a full-fledged installer is planned, which will
    eliminate this 'quirk'.

Project workflow with AutoAtom's GUI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There is an implicit logical progression when analyzing images with the AutoAtom GUI. When working with images, the
project file will be in certain *stages*, and what state the project is in will affect what you can and/or should do
next. The stages are

    #. A dm3 HAADF-STEM image has been imported as a project file, but no other analysis has taken place yet. the project is in an 'initial' state.

    #. Column detection has been performed, and the project now has information about 2D atomic positions. The project is now in a 'column' state.

    #. A spatial mapping algorithm has been performed, so each column now has information about the location of its 8 closest neighbours. The project is in 'spatial map' state.

    #. Column characterization has been applied, and colums now have information about the probability of its own atomic species, its z-position, its neighbours in the opposite and same crystal plane, etc... The project is in a 'result' stage.

    #. Manual consideration of the data, and manual corrections and control has been performed by the user. This is the final state, and the project in now in a 'control' state.

It is the 'control' state that one would use to analyse data, perform principal component analysis, generate plots
and/or export data. It is important to note though, that these 'states' are only implicit, and is not internally
tracked, and even though the GUI has checks in place to make sure invalid operations are not performed, some of the
software's methods assume a certain state, but can be performed in other states as well, with possibly unpredictable
results. The outline given below, should give a feel for how the GUI is intended to be used.

Initial stage
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

Before importing an .dm3 -file into AutoAtom, one will usually find it beneficial to prepare the image in a specific
way. Using a program such as digital micrograph (DMG), one should apply a fft-mask to reduce the noise in the image. In
addition, one could use digital micrograph's scaling algorithm to upscale the image if it is small and/or low
magnification (The scale should typically not be any lower than \~6 pm / pixel). These preparation steps will greatly
increase the effectiveness of the column detection algorithm. In the future, these techniques might be included directly
in the software, but for now, pre-processing in DMG is necessary.

.. Note::

    The filetype of .dm3 must be maintained. It is the only file-type supported for import, and contains essential
    metadata. For example, when rescaled in DMG, the 'scale' field of the dm3 metadata is correctly and automatically
    updated.

Now that we have a pre-processed .dm3 file ready, we can open AutoAtom, and from the 'file' menu select 'new'. Using the
file-dialog, locate the .dm3 and hit 'open'.

With the program there is a sample image included which can be used to get familiar with the software. This file is
called 'sample.dm3', and is already pre-processed, so can be imported directly. Once 'sample.dm3' is imported, the
project instance that the GUI creates, is now in the 'initial' state, and can now be saved as a project file using
'file -> save'. Typically though, one would proceed directly to column detection from this state.

Column stage
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

We now wish to locate the positions of columns in the image.




Exporting data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Generating plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Performing built-in principle component analysis (PCA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Writing plugins for AutoAtom6000
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


