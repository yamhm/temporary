import pandas as pd

# Specify the file paths in Google Drive
tutors_file = 'tutors_generated.xlsx'
students_file = 'students_generated.xlsx'

# Load the tutor data
tutors_df = pd.read_excel(tutors_file, sheet_name=0)

# Load the student data
students_df = pd.read_excel(students_file, sheet_name=0)

# Display the first few rows of each dataframe
print("Tutors Data:")
print(tutors_df.head())

print("\nStudents Data:")
print(students_df.head())

import torch
import pandas as pd
import json
from tqdm import tqdm

def preprocess_availability_gpu(tutors_df, students_df):
    """
    Preprocesses availability and compatibility between tutors and students using GPU acceleration.

    Parameters:
        tutors_df (pd.DataFrame): Dataframe containing tutor information, including 'Availability'.
        students_df (pd.DataFrame): Dataframe containing student information, including 'Availability'.

    Returns:
        compatibility_matrix (torch.Tensor): Binary tensor indicating compatibility between students and tutors.
    """
    max_time_slots = 57  # Adjust based on the maximum number of time slots across tutors/students

    # Parse tutor availability into binary tensors
    tutor_availability_tensor = torch.zeros((len(tutors_df), max_time_slots), dtype=torch.bool, device='cuda')
    for i, avail in enumerate(tutors_df['Availability']):
        try:
            slots = list(json.loads(avail).values()) if isinstance(avail, str) else avail
            if slots:
                tutor_availability_tensor[i, slots] = True
        except (json.JSONDecodeError, TypeError):
            continue

    # Parse student availability into binary tensors
    student_availability_tensor = torch.zeros((len(students_df), max_time_slots), dtype=torch.bool, device='cuda')
    for i, student in students_df.iterrows():
        try:
            slots = eval(student['Availability']) if isinstance(student['Availability'], str) else student['Availability']
            if slots:
                student_availability_tensor[i, slots] = True
        except (SyntaxError, ValueError):
            continue

    # Compute compatibility using matrix multiplication
    compatibility_matrix = torch.mm(student_availability_tensor.float(), tutor_availability_tensor.float().T) > 0

    return compatibility_matrix


def vcg_auction_pytorch_with_matrix(tutors_df, students_df, compatibility_matrix):
    """
    Conducts a VCG auction using PyTorch for GPU acceleration, leveraging a compatibility matrix.

    Parameters:
        tutors_df (pd.DataFrame): Dataframe containing tutor information.
        students_df (pd.DataFrame): Dataframe containing student information.
        compatibility_matrix (torch.Tensor): Binary tensor indicating compatibility between students and tutors.

    Returns:
        allocation (dict): A dictionary with student IDs as keys and the assigned tutor as values.
        payments (dict): A dictionary with student IDs as keys and the payments made as values.
    """
    allocation = {}
    payments = {}

    # Ensure numeric data in tutors_df
    numeric_columns = ['rating', 'reviews_number', 'lessons_number']
    language_columns = [col for col in tutors_df.columns if col.startswith("language_")]

    tutors_tensor = torch.tensor(
        tutors_df[numeric_columns + language_columns].to_numpy(), dtype=torch.float32
    ).to('cuda')  # Move to GPU

    # Precompute masks for Arabic special case
    language_mask = torch.all(
        torch.tensor((tutors_df[language_columns] == 0).to_numpy(), dtype=torch.bool),
        dim=1
    ).to('cuda')  # Move to GPU

    # Iterate over students with a progress bar
    for student_idx, student in tqdm(enumerate(students_df.itertuples()), total=len(students_df), desc="VCG Auction"):
        # Parse desired subjects
        try:
            desired_subjects = eval(student.Desired_subjects) if isinstance(student.Desired_subjects, str) else student.Desired_subjects
        except (SyntaxError, ValueError):
            continue

        if not desired_subjects or not isinstance(desired_subjects, (list, set)):
            continue

        # Parse w5
        try:
            w5_dict = json.loads(student.w5) if isinstance(student.w5, str) else student.w5
        except (json.JSONDecodeError, TypeError):
            continue

        # Extract student-specific weights
        w2, w3, w4, w6 = torch.tensor([student.w2, student.w3, student.w4, student.w6], dtype=torch.float32).to('cuda')

        # Precompute utilities for compatible tutors
        utilities = torch.zeros(tutors_tensor.size(0), device='cuda')
        compatible_tutors_mask = compatibility_matrix[student_idx]

        for subject in desired_subjects:
            if subject == "Arabic":
                eligible_mask = language_mask
            else:
                subject_column = f"language_{subject}"
                if subject_column not in tutors_df.columns:
                    continue
                eligible_mask = torch.tensor(
                    tutors_df[subject_column].to_numpy() == 1, dtype=torch.bool
                ).to('cuda')

            # Combine compatibility and eligibility masks
            eligible_mask &= compatible_tutors_mask

            # Add w5 value for the subject
            w5_value = torch.tensor(w5_dict.get(subject, 0), dtype=torch.float32).to('cuda')

            # Compute utility for eligible tutors
            subject_utility = (
                w2 * tutors_tensor[:, 0] +  # rating
                w3 * tutors_tensor[:, 1] +  # reviews_number
                w4 * tutors_tensor[:, 2] +  # lessons_number
                w5_value +  # w5 value
                w6  # constant factor
            )

            # Apply mask for eligibility
            utilities += subject_utility * eligible_mask

        # Assign the tutor with the highest utility
        if utilities.max().item() > 0:
            best_tutor_idx = torch.argmax(utilities).item()
            allocation[student.Index] = tutors_df.iloc[best_tutor_idx]['name']

            # VCG payment calculation
            utilities[best_tutor_idx] = 0  # Remove the highest utility tutor
            payments[student.Index] = utilities.max().item()  # Second-best utility

    return allocation, payments

from tqdm import tqdm

def gale_shapley_gpu_chunked(tutors_df, students_df, compatibility_matrix, chunk_size=1000):
    """
    Implements the Gale-Shapley algorithm for stable matching in a two-sided market using GPU acceleration
    with chunked processing to manage memory usage.

    Parameters:
        tutors_df (pd.DataFrame): Dataframe containing tutor information.
        students_df (pd.DataFrame): Dataframe containing student information.
        compatibility_matrix (torch.Tensor): Binary tensor indicating compatibility between students and tutors.
        chunk_size (int): Number of students to process in each chunk to reduce memory usage.

    Returns:
        student_allocation (torch.Tensor): Tensor with student indices as rows and matched tutor indices as values.
        tutor_allocation (torch.Tensor): Tensor with tutor indices as rows and matched student indices as values.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    num_students = len(students_df)
    num_tutors = len(tutors_df)

    # Initialize allocations
    student_allocation = torch.full((num_students,), -1, dtype=torch.long, device=device)
    tutor_allocation = torch.full((num_tutors,), -1, dtype=torch.long, device=device)

    # Free students
    free_students = torch.ones(num_students, dtype=torch.bool, device=device)

    # Track proposals as a binary matrix
    proposals = torch.zeros((num_students, num_tutors), dtype=torch.bool, device=device)

    # Use half precision for utilities
    compatibility_matrix = compatibility_matrix.to(device).to(torch.float16)
    student_utilities = compatibility_matrix.clone() * torch.rand_like(compatibility_matrix, dtype=torch.float16, device=device)
    tutor_utilities = compatibility_matrix.T.clone() * torch.rand_like(compatibility_matrix.T, dtype=torch.float16, device=device)

    # Progress bar
    progress_bar = tqdm(total=num_students, desc="Processing Free Students")

    while free_students.any():
        # Get free students in chunks
        free_indices = torch.where(free_students)[0]
        for chunk_start in range(0, len(free_indices), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(free_indices))
            chunk_indices = free_indices[chunk_start:chunk_end]

            # Compute masked utilities for the chunk of free students
            masked_utilities = student_utilities[chunk_indices].clone()
            masked_utilities[proposals[chunk_indices]] = -float('inf')  # Mask already proposed tutors

            # Each free student in the chunk proposes to their most preferred compatible tutor
            best_tutors = torch.argmax(masked_utilities, dim=1)
            proposals[chunk_indices, best_tutors] = True

            for i, student_idx in enumerate(chunk_indices):
                tutor_idx = best_tutors[i]

                # Check if the tutor is free
                if tutor_allocation[tutor_idx] == -1:
                    # Match student and tutor
                    tutor_allocation[tutor_idx] = student_idx
                    student_allocation[student_idx] = tutor_idx
                    free_students[student_idx] = False
                else:
                    # Compare current match and new proposal
                    current_student = tutor_allocation[tutor_idx]
                    current_utility = tutor_utilities[tutor_idx, current_student]
                    new_utility = tutor_utilities[tutor_idx, student_idx]

                    if new_utility > current_utility:
                        # Replace current student with new student
                        tutor_allocation[tutor_idx] = student_idx
                        student_allocation[student_idx] = tutor_idx
                        free_students[student_idx] = False
                        free_students[current_student] = True  # Free the current student
                    else:
                        # Keep current student; student remains free
                        free_students[student_idx] = True

            # Free GPU memory for the processed chunk
            del masked_utilities
            torch.cuda.empty_cache()

        # Update the progress bar
        progress_bar.update(len(free_indices))

    # Close the progress bar
    progress_bar.close()

    return student_allocation, tutor_allocation

# Preprocess compatibility matrix with GPU acceleration
compatibility_matrix = preprocess_availability_gpu(tutors_df, students_df)

# Run the Gale-Shapley algorithm with matrix-based GPU acceleration
student_allocation, tutor_allocation = gale_shapley_gpu_chunked(
    tutors_df, students_df, compatibility_matrix, chunk_size=500
)

print("Student Allocation (Tensor):", student_allocation)
print("Tutor Allocation (Tensor):", tutor_allocation)

import pickle

def save_and_evaluate_allocations(
    tutors_df, students_df, student_allocation, tutor_allocation, compatibility_matrix
):
    """
    Saves the final allocations and calculates statistics for total and average utilities,
    and the number of matched and unmatched pairs.

    Parameters:
        tutors_df (pd.DataFrame): Dataframe containing tutor information.
        students_df (pd.DataFrame): Dataframe containing student information.
        student_allocation (torch.Tensor): Tensor with student indices and matched tutor indices.
        tutor_allocation (torch.Tensor): Tensor with tutor indices and matched student indices.
        compatibility_matrix (torch.Tensor): Binary tensor indicating compatibility between students and tutors.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Calculate total and average utility for students
    student_utilities = compatibility_matrix[torch.arange(len(students_df), device=device), student_allocation]
    total_student_utility = student_utilities[student_utilities > 0].sum().item()
    average_student_utility = student_utilities[student_utilities > 0].mean().item()

    # Calculate total and average utility for tutors
    tutor_utilities = compatibility_matrix.T[torch.arange(len(tutors_df), device=device), tutor_allocation]
    total_tutor_utility = tutor_utilities[tutor_utilities > 0].sum().item()
    average_tutor_utility = tutor_utilities[tutor_utilities > 0].mean().item()

    # Calculate matched and unmatched pairs
    num_matched_students = (student_allocation != -1).sum().item()
    num_unmatched_students = (student_allocation == -1).sum().item()
    num_matched_tutors = (tutor_allocation != -1).sum().item()
    num_unmatched_tutors = (tutor_allocation == -1).sum().item()

    # Prepare allocation data for saving
    allocation_df = pd.DataFrame({
        'Student_ID': students_df.index,
        'Matched_Tutor_ID': student_allocation.cpu().numpy()
    })
    allocation_df['Matched_Tutor_Name'] = allocation_df['Matched_Tutor_ID'].apply(
        lambda idx: tutors_df.iloc[idx]['name'] if idx != -1 else "Unmatched"
    )

    # Save to CSV
    allocation_df.to_csv('final_allocations.csv', index=False)

    # Save to Pickle
    with open('final_allocations.pkl', 'wb') as pkl_file:
        pickle.dump(allocation_df, pkl_file)

    # Display results
    print("Total Student Utility:", total_student_utility)
    print("Average Student Utility:", average_student_utility)
    print("Total Tutor Utility:", total_tutor_utility)
    print("Average Tutor Utility:", average_tutor_utility)
    print("Number of Matched Students:", num_matched_students)
    print("Number of Unmatched Students:", num_unmatched_students)
    print("Number of Matched Tutors:", num_matched_tutors)
    print("Number of Unmatched Tutors:", num_unmatched_tutors)

    # Return statistics for further use
    return {
        'total_student_utility': total_student_utility,
        'average_student_utility': average_student_utility,
        'total_tutor_utility': total_tutor_utility,
        'average_tutor_utility': average_tutor_utility,
        'num_matched_students': num_matched_students,
        'num_unmatched_students': num_unmatched_students,
        'num_matched_tutors': num_matched_tutors,
        'num_unmatched_tutors': num_unmatched_tutors
    }


# Evaluate and save results
results = save_and_evaluate_allocations(
    tutors_df, students_df, student_allocation, tutor_allocation, compatibility_matrix
)

print("Results:", results)
