import '@tensorflow/tfjs-backend-cpu';
import * as tflite from '@tensorflow/tfjs-tflite';
import Camera, { FACING_MODES } from 'react-html5-camera-photo';
import 'react-html5-camera-photo/build/css/index.css';
import {useCallback} from "react";

tflite.setWasmPath('tflite_wasm/');

function App() {

    const onTakePhoto = useCallback(() => {

    }, [])

    return (
        <div className={'wrap'}>

            <Camera
                idealFacingMode={FACING_MODES.ENVIRONMENT}
                onTakePhoto={onTakePhoto}
                isSilentMode
                isFullscreen
            />
        </div>
    );
}

export default App;
