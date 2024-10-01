from tensorflow.keras import layers, Model
from models.image_model import create_image_model
from models.genomics_model import create_genomic_model
from models.ehr_model import create_ehr_model


def create_combined_model():
    # Image model
    image_model = create_image_model()
    image_input = image_model.input
    image_output = image_model.output

    # Genomic model
    genomic_input = layers.Input(shape=(1000,))  # Adjust input shape as per the genomic data
    genomic_model = create_genomic_model(1000)
    genomic_output = genomic_model(genomic_input)

    # EHR model
    ehr_input = layers.Input(shape=(200,))  # Adjust input shape as per EHR data
    ehr_model = create_ehr_model(200)
    ehr_output = ehr_model(ehr_input)

    # Concatenate the outputs from each model
    concatenated = layers.concatenate([image_output, genomic_output, ehr_output])

    # Final dense layers after concatenation
    dense = layers.Dense(64, activation='relu')(concatenated)
    dense = layers.Dense(1, activation='sigmoid')(dense)

    # Create and compile the final model
    model = Model(inputs=[image_input, genomic_input, ehr_input], outputs=dense)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
if __name__ == '__main__':
    model = create_combined_model()
    model.summary()
